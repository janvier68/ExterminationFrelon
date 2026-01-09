#!/bin/bash
set -euo pipefail

# Fusionne dossier2 -> dossier1 sans écraser : en cas de collision, renomme avec _dupN
if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <dossier1_destination> <dossier2_source>"
  exit 1
fi

DEST_ROOT="$(realpath "$1")"
SRC_ROOT="$(realpath "$2")"


if [[ ! -d "$DEST_ROOT" ]]; then
  echo "Erreur: dossier1 n'existe pas: $DEST_ROOT" >&2
  exit 1
fi
if [[ ! -d "$SRC_ROOT" ]]; then
  echo "Erreur: dossier2 n'existe pas: $SRC_ROOT" >&2
  exit 1
fi
if [[ "$DEST_ROOT" == "$SRC_ROOT" ]]; then
  echo "Erreur: les deux dossiers sont identiques." >&2
  exit 1
fi


###############################################
# 1) Nettoyage dataset sauf .zip
###############################################

# Supprimer tout sauf les .zip
find "$DEST_ROOT" -mindepth 1 ! -name "*.zip" -exec rm -rf {} \;

###############################################
# 2) Dézip
###############################################

ZIP_FILE=$(find "$DEST_ROOT" -maxdepth 1 -name "*.zip" | head -n 1)
unzip "$ZIP_FILE" -d "$DEST_ROOT"

###############################################
# 3) Fusion datasets (code intégré)
###############################################



shopt -s nullglob dotglob

splits=(train valid test)
kinds=(images labels)

unique_target() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    printf '%s' "$target"
    return 0
  fi
  local dir base name ext n
  dir="$(dirname "$target")"
  base="$(basename "$target")"
  name="${base%.*}"
  ext=""
  [[ "$base" == *.* ]] && ext=".${base##*.}"

  n=1
  local cand
  while : ; do
    cand="$dir/${name}_dup${n}${ext}"
    [[ ! -e "$cand" ]] && { printf '%s' "$cand"; return 0; }
    ((n++))
  done
}

move_with_rename() {
  local src_file="$1"
  local dest_dir="$2"
  mkdir -p "$dest_dir"
  local base target unique
  base="$(basename "$src_file")"
  target="$dest_dir/$base"
  unique="$(unique_target "$target")"
  mv "$src_file" "$unique"
}

echo "Fusion de: $SRC_ROOT -> $DEST_ROOT"
echo

for split in "${splits[@]}"; do
  for kind in "${kinds[@]}"; do
    src_dir="$SRC_ROOT/$split/$kind"
    dest_dir="$DEST_ROOT/$split/$kind"

    if [[ -d "$src_dir" ]]; then
      mkdir -p "$dest_dir"
      echo "-> $split/$kind :"
      moved_any=false
      for item in "$src_dir"/*; do
        [[ -e "$item" ]] || continue
        if [[ -f "$item" ]]; then
          move_with_rename "$item" "$dest_dir"
          moved_any=true
        elif [[ -d "$item" ]]; then
          find "$item" -type f -print0 | while IFS= read -r -d '' f; do
            move_with_rename "$f" "$dest_dir"
            moved_any=true
          done
          rm -rf "$item"
        fi
      done
      rmdir --ignore-fail-on-non-empty "$src_dir" 2>/dev/null || true
      if ! $moved_any; then
        echo "   (rien à déplacer)"
      fi
    fi
  done
done

find "$SRC_ROOT" -type d -empty -delete 2>/dev/null || true

echo
echo "Fusion terminée."
