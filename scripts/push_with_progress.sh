#!/usr/bin/env bash
# push_with_progress.sh — pousse de gros volumes vers origin en commits découpés
# (~900 Mo max par commit) avec une barre de progression temps réel.
#
# Usage :
#   bash scripts/push_with_progress.sh [--dry-run] <dossier> [dossier...]
#   bash scripts/push_with_progress.sh [--dry-run] -- <paths...>
#
# Authentification : si GIT_PAT est défini, il est utilisé comme token GitHub
# (jamais écrit sur disque). Sinon, les credentials git classiques s'appliquent.
#
# Variables d'environnement :
#   MAX_CHUNK_BYTES  taille max d'un commit (défaut 900 Mo)
#   PUSH_RETRIES     nombre d'essais par morceau (défaut 8)
#   REMOTE / BRANCH  cible du push (défauts : origin / main)
set -u

# Taille lisible : Go au-delà de 1 Go, Mo sinon
fmt_size() {
  awk -v b="$1" 'BEGIN{ if (b >= 1073741824) printf "%.2f Go", b/1073741824; else printf "%.1f Mo", b/1048576 }'
}

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$REPO_ROOT" || exit 1

MAX_BYTES=${MAX_CHUNK_BYTES:-$((900 * 1024 * 1024))}
RETRIES=${PUSH_RETRIES:-8}
REMOTE=${REMOTE:-origin}
BRANCH=${BRANCH:-main}
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && { DRY_RUN=1; shift; }
[ $# -eq 0 ] && { echo "Usage: bash $0 [--dry-run] <dossier> [dossier...]"; exit 1; }

# Askpass créé à la volée : lit le token depuis l'environnement, rien sur disque.
if [ -n "${GIT_PAT:-}" ]; then
  printf '#!/bin/sh\ncase "$1" in\n  *[Uu]sername*) echo x-access-token ;;\n  *) echo "$GIT_PAT" ;;\nesac\n' > /tmp/git-askpass.sh
  chmod +x /tmp/git-askpass.sh
  export GIT_ASKPASS=/tmp/git-askpass.sh
fi
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=90

TMPD=$(mktemp -d /tmp/push_progress.XXXXXX)
trap 'printf "\n"; rm -rf "$TMPD"' EXIT

# ---------------------------------------------------------------------------
# Barre de progression (une seule ligne, mise à jour en \r)
#   frac : avancement global [0..1]
#   info : texte libre affiché après le pourcentage
# ---------------------------------------------------------------------------
render() {
  local frac=$1 info=$2
  awk -v f="$frac" -v info="$info" 'BEGIN{
    w=32; n=int(f*w+0.5); if(n<0)n=0; if(n>w)n=w;
    printf "\r\033[2K[";
    for(i=0;i<w;i++) printf (i<n ? "\033[32m#\033[0m" : "-");
    printf "] %5.1f%%  %s", f*100, info;
  }'
}

# ---------------------------------------------------------------------------
# Manifest : découpe chaque dossier en morceaux <= MAX_BYTES
# Fichier manifest : octets<TAB>liste<TAB>label
# ---------------------------------------------------------------------------
MANIFEST="$TMPD/manifest.txt"
TOTAL_BYTES=0
N_PARTS=0

for dir in "$@"; do
  [ -d "$dir" ] || { echo "⚠ dossier introuvable: $dir"; exit 1; }
  # Fichiers du dossier, moins les fichiers ignorés par git (cache, secrets, ...)
  find "$dir" -type f \
    ! -path '*/.ipynb_checkpoints/*' ! -path '*/__pycache__/*' \
    ! -name '.DS_Store' ! -name '*.pyc' \
    | sort > "$TMPD/all.txt"
  git check-ignore --stdin < "$TMPD/all.txt" > "$TMPD/ignored.txt" 2>/dev/null || true
  comm -23 "$TMPD/all.txt" "$TMPD/ignored.txt" > "$TMPD/kept.txt"
  : > "$TMPD/sizes.txt"
  while IFS= read -r p; do
    printf '%s\t%s\n' "$(stat -c '%s' "$p" 2>/dev/null || echo 0)" "$p" >> "$TMPD/sizes.txt"
  done < "$TMPD/kept.txt"
  [ -s "$TMPD/sizes.txt" ] || { echo "⚠ rien à pousser dans: $dir"; continue; }
  # Découpe par accumulation d'octets : part_<i>.txt (fichiers) + part_bytes.txt (tailles)
  awk -F'\t' -v max="$MAX_BYTES" -v d="$TMPD" 'BEGIN { i = 0 }
  {
    if (acc + $1 > max && acc > 0) { i++; acc = 0 }
    acc += $1
    print $2 > (d "/part_" i ".txt")
    part_bytes[i] = acc
  } END {
    for (k = 0; k <= i; k++) print part_bytes[k] > (d "/part_bytes.txt")
  }' "$TMPD/sizes.txt"
  nparts=$(wc -l < "$TMPD/part_bytes.txt")
  for i in $(seq 1 "$nparts"); do
    local_bytes=$(sed -n "${i}p" "$TMPD/part_bytes.txt")
    local_human=$(fmt_size "$local_bytes")
    printf '%s\t%s\t%s\n' "$local_bytes" "$TMPD/part_$((i-1)).txt" "$dir #$((i-1)) ($local_human)" >> "$MANIFEST"
    TOTAL_BYTES=$((TOTAL_BYTES + local_bytes))
    N_PARTS=$((N_PARTS + 1))
  done
done

[ "$N_PARTS" -gt 0 ] || { echo "Rien à pousser."; exit 0; }
echo "Manifest: $N_PARTS morceaux, $(fmt_size "$TOTAL_BYTES") au total."
[ "$DRY_RUN" -eq 1 ] && { awk -F'\t' '{printf "  %s\n", $3}' "$MANIFEST"; exit 0; }

# ---------------------------------------------------------------------------
# Pousse chaque morceau : commit puis push avec essais et vérification
# ---------------------------------------------------------------------------
git reset -q
DONE_BYTES=0
DONE_PARTS=0
START_TS=$(date +%s)

while IFS=$'\t' read -r bytes list label; do
  DONE_PARTS=$((DONE_PARTS + 1))
  git add --pathspec-from-file="$list" 2>/dev/null || { echo "Erreur add: $label"; exit 1; }
  n=$(git diff --cached --name-only | wc -l)
  if [ "$n" -gt 0 ]; then
    git commit -q -m "pousse: $label" || { echo "Erreur commit: $label"; exit 1; }
  fi
  sha=$(git rev-parse HEAD)
  ok=0
  try=1
  while [ "$try" -le "$RETRIES" ]; do
    : > "$TMPD/push.log"
    ( git push --progress "$REMOTE" "$sha":refs/heads/"$BRANCH" > "$TMPD/push.log" 2>&1 ) &
    pid=$!
    # Suivi temps réel : parse le % de git + met à jour la barre globale
    while kill -0 "$pid" 2>/dev/null; do
      gpct=$(grep -oE 'Writing objects: *[0-9]+%' "$TMPD/push.log" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
      gpct=${gpct:-0}
      frac=$(awk -v done="$DONE_BYTES" -v cur="$bytes" -v p="$gpct" -v tot="$TOTAL_BYTES" \
        'BEGIN{ if (tot>0) printf "%.4f", (done + cur*p/100) / tot; else print 0 }')
      render "$frac" "morceau $DONE_PARTS/$N_PARTS — $label — transfert $gpct% (essai $try)"
      sleep 1
    done
    wait "$pid" && rc=0 || rc=$?
    if [ "$rc" -eq 0 ]; then
      remote_sha=$(git ls-remote "$REMOTE" refs/heads/"$BRANCH" 2>/dev/null | awk '{print $1}')
      if [ "$remote_sha" = "$sha" ]; then ok=1; break; fi
    fi
    try=$((try + 1))
    sleep 20
  done
  [ "$ok" -eq 1 ] || { render "$(awk -v d=$DONE_BYTES -v t=$TOTAL_BYTES 'BEGIN{printf "%.4f", (t>0?d/t:0)}')" "ÉCHEC: $label"; echo; exit 1; }
  DONE_BYTES=$((DONE_BYTES + bytes))
  render "$(awk -v d=$DONE_BYTES -v t=$TOTAL_BYTES 'BEGIN{printf "%.4f", (t>0?d/t:0)}')" "morceau $DONE_PARTS/$N_PARTS ✓ — $label"
  echo
  sleep 10
done < "$MANIFEST"

ELAPSED=$(( $(date +%s) - START_TS ))
echo ""
echo "✓ Terminé : $N_PARTS morceaux poussés sur $REMOTE/$BRANCH en $((ELAPSED / 60)) min $((ELAPSED % 60)) s."
