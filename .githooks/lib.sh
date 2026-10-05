# Shared helpers for the git hooks. Sourced, not executed.

REPO_ROOT="$(git rev-parse --show-toplevel)"
CORE_PATHS_FILE="$REPO_ROOT/.githooks/core_paths"

# The core path entries, without comments, blank lines or CRs.
core_entries() {
    sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e 's/^[[:space:]]*//' \
        "$CORE_PATHS_FILE" | tr -d '\r' | grep -v '^$'
}

# is_core PATH: true if PATH (repo-relative) is core.
is_core() {
    local entry
    while IFS= read -r entry; do
        case "$entry" in
            */) [[ "$1" == "$entry"* ]] && return 0 ;;
            *)  [[ "$1" == "$entry" ]] && return 0 ;;
        esac
    done < <(core_entries)
    return 1
}

# Staged files that are core, one per line. --no-renames lists both sides of a
# rename, so moving a file out of core still counts as touching core.
staged_core_files() {
    git diff --cached --name-only --no-renames -z |
        while IFS= read -r -d '' f; do
            is_core "$f" && printf '%s\n' "$f"
        done
}
