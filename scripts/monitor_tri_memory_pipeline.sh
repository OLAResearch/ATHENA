#!/usr/bin/env bash

# Idempotent half-hour LUMI monitor for the corrected seven-way tri-memory
# experiment.  It submits only the next stage after its prerequisites pass,
# never cancels jobs, and copies terminal logs/artifacts locally.

set -uo pipefail

PROJECT_ROOT=/Users/mingyl/ATHENA
REMOTE_HOST=lumi.csc.fi
REMOTE_ROOT=/scratch/project_462001493/mingyl/ATHENA
REMOTE_CACHE_ROOT=/scratch/project_462001050
PIPELINE_TAG=tri_memory_joint_softsubset_20260903T053716Z
SOURCE_TAG=tri_memory_joint_softsubset_20260903T090500Z
REMOTE_SNAPSHOT=$REMOTE_ROOT/snapshots/$PIPELINE_TAG
REMOTE_SCRIPT_ROOT=$REMOTE_ROOT/scripts/$PIPELINE_TAG
REMOTE_RUN_ROOT=$REMOTE_ROOT/run/$PIPELINE_TAG
REMOTE_STATE_ROOT=$REMOTE_RUN_ROOT/state
REMOTE_BASELINE_ROOT=$REMOTE_ROOT/run/$SOURCE_TAG/engram_20m_train
LOCAL_LOG_ROOT=$PROJECT_ROOT/logs/csc/lumi/scheduler
LOCAL_JOB_LOG_ROOT=$PROJECT_ROOT/logs/csc/lumi
LOG_FILE=$LOCAL_LOG_ROOT/tri_memory_monitor.log
REPORT_FILE=$LOCAL_LOG_ROOT/tri_memory_pipeline_report.md
LOCK_DIR=$LOCAL_LOG_ROOT/.tri-memory-lock
AUDIT_SCRIPT=$REMOTE_CACHE_ROOT/mingyl/PathhunterPlus/.codex/skills/lumi-minimize-gpu-hours/scripts/audit_sbatch.py

mkdir -p "$LOCAL_LOG_ROOT" "$LOCAL_JOB_LOG_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

timestamp() {
  /bin/date -u '+%Y-%m-%dT%H:%M:%SZ'
}

log() {
  /usr/bin/printf '[%s] %s\n' "$(timestamp)" "$*" >> "$LOG_FILE"
}

remote() {
  /usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=2 "$REMOTE_HOST" "$@"
}

remote_value() {
  local path=$1
  remote "if test -f '$path'; then /bin/cat '$path'; fi" 2>/dev/null \
    | /usr/bin/tr -d '\r\n '
}

remote_mark() {
  local path=$1 value=$2
  remote "printf '%s\\n' '$value' > '$path'" >/dev/null
}

remote_has() {
  remote "test -e '$1'"
}

remote_file_contains() {
  remote "/usr/bin/grep -F -q -- '$2' '$1'"
}

queue_snapshot() {
  log 'squeue before monitor/submission:'
  remote "/usr/bin/squeue -u mingyl -o '%.18i %.12P %.34j %.10T %.12M %.12l %R'" \
    >> "$LOG_FILE" 2>&1 || log 'WARNING: unable to read squeue'
}

job_state() {
  local jid=$1 state
  state=$(remote "/usr/bin/sacct -X -n -P -j '$jid' --format=State | /usr/bin/head -1 | /usr/bin/tr -d '[:space:]'" 2>/dev/null || true)
  if [ -z "$state" ]; then
    state=$(remote "/usr/bin/squeue -h -j '$jid' -o '%T' | /usr/bin/head -1 | /usr/bin/tr -d '[:space:]'" 2>/dev/null || true)
  fi
  /usr/bin/printf '%s' "$state"
}

state_is_success() {
  [ "$1" = COMPLETED ]
}

state_is_failure() {
  case "$1" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|DEPENDENCYNEVER_SATISFIED|DEPENDENCY_NEVER_SATISFIED)
      return 0 ;;
    *) return 1 ;;
  esac
}

stage_jid() {
  remote_value "$REMOTE_STATE_ROOT/$1.jobid"
}

stage_failed() {
  remote_has "$REMOTE_STATE_ROOT/$1.failed"
}

stage_log_path() {
  local stage=$1 jid=$2
  case "$stage" in
    smoke) /usr/bin/printf '%s/logs/tri-smoke-%s.out' "$REMOTE_ROOT" "$jid" ;;
    train) /usr/bin/printf '%s/logs/tri-memory-train-%s.out' "$REMOTE_ROOT" "$jid" ;;
    eval_*) /usr/bin/printf '%s/logs/tri-eval-%s.out' "$REMOTE_ROOT" "$jid" ;;
    bootstrap) /usr/bin/printf '%s/logs/tri-memory-stats-%s.out' "$REMOTE_ROOT" "$jid" ;;
    oracle) /usr/bin/printf '%s/logs/tri-oracle-fix-%s.out' "$REMOTE_ROOT" "$jid" ;;
    *) return 1 ;;
  esac
}

stage_err_path() {
  local out
  out=$(stage_log_path "$1" "$2") || return 1
  /usr/bin/printf '%s' "${out%.out}.err"
}

stage_result_path() {
  local stage=$1 task
  case "$stage" in
    train) /usr/bin/printf '%s/tri_20m_train/results.json' "$REMOTE_RUN_ROOT" ;;
    eval_*) task=${stage#eval_}; /usr/bin/printf '%s/%s_tri_full_eval/results.json' "$REMOTE_RUN_ROOT" "$task" ;;
    bootstrap) /usr/bin/printf '%s/nq_tri_full_eval/results_with_bootstrap.json' "$REMOTE_RUN_ROOT" ;;
    oracle) /usr/bin/printf '%s/nq_tri_full_eval/results_with_corrected_oracle.json' "$REMOTE_RUN_ROOT" ;;
    *) return 0 ;;
  esac
}

copy_remote_file() {
  local remote_path=$1 local_path=$2
  if remote_has "$remote_path"; then
    /bin/mkdir -p "$(/usr/bin/dirname "$local_path")"
    /usr/bin/scp -q -o BatchMode=yes -o ConnectTimeout=20 \
      "$REMOTE_HOST:$remote_path" "$local_path" >> "$LOG_FILE" 2>&1 \
      || log "WARNING: failed to copy $remote_path"
  fi
}

sync_stage_logs() {
  local stage=$1 jid=$2 destination=$LOCAL_JOB_LOG_ROOT/$jid
  if [ -f "$destination/.downloaded" ]; then
    return 0
  fi
  /bin/mkdir -p "$destination"
  copy_remote_file "$(stage_log_path "$stage" "$jid")" "$destination/stdout.out"
  copy_remote_file "$(stage_err_path "$stage" "$jid")" "$destination/stderr.err"
  case "$stage" in
    train)
      copy_remote_file "$REMOTE_RUN_ROOT/tri_20m_train/results.json" "$destination/results.json"
      copy_remote_file "$REMOTE_RUN_ROOT/tri_20m_train/config.json" "$destination/config.json" ;;
    eval_*)
      local task=${stage#eval_}
      copy_remote_file "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results.json" "$destination/results.json" ;;
    bootstrap)
      for task in nq webqa triviaqa truthfulqa hotpotqa; do
        copy_remote_file "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results_with_bootstrap.json" \
          "$destination/${task}_results_with_bootstrap.json"
      done ;;
    oracle)
      for task in nq webqa triviaqa truthfulqa hotpotqa; do
        copy_remote_file "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results_with_corrected_oracle.json" \
          "$destination/${task}_results_with_corrected_oracle.json"
      done ;;
  esac
  /usr/bin/touch "$destination/.downloaded"
}

ensure_remote_layout() {
  if remote_has "$REMOTE_STATE_ROOT/snapshot_ready"; then
    return 0
  fi
  log "Preparing isolated snapshot $PIPELINE_TAG"
  remote "/bin/mkdir -p '$REMOTE_SNAPSHOT' '$REMOTE_SCRIPT_ROOT' '$REMOTE_RUN_ROOT' '$REMOTE_STATE_ROOT' '$REMOTE_ROOT/logs' '$REMOTE_RUN_ROOT/engram_20m_train'" || return 1
  /usr/bin/rsync -a --exclude '__pycache__' --exclude '.pytest_cache' \
    "$PROJECT_ROOT/engram/" "$REMOTE_HOST:$REMOTE_SNAPSHOT/engram/" >> "$LOG_FILE" 2>&1 || return 1
  /usr/bin/rsync -a --exclude '__pycache__' --exclude '.pytest_cache' \
    "$PROJECT_ROOT/scripts/" "$REMOTE_HOST:$REMOTE_SNAPSHOT/scripts/" >> "$LOG_FILE" 2>&1 || return 1
  local script
  for script in lumi_tri_memory_smoke.slurm lumi_tri_memory_joint.slurm \
    lumi_tri_memory_eval.slurm lumi_tri_memory_bootstrap.slurm \
    lumi_tri_memory_corrected_oracle.slurm; do
    /usr/bin/rsync -a "$PROJECT_ROOT/run/$script" \
      "$REMOTE_HOST:$REMOTE_SCRIPT_ROOT/$script" >> "$LOG_FILE" 2>&1 || return 1
  done
  remote "for f in adaptor_best.pt config.json; do if ! test -e '$REMOTE_RUN_ROOT/engram_20m_train/'\$f; then cp '$REMOTE_BASELINE_ROOT/'\$f '$REMOTE_RUN_ROOT/engram_20m_train/'\$f; fi; done" || return 1
  remote "/bin/chmod +x '$REMOTE_SCRIPT_ROOT'/*.slurm && /bin/touch '$REMOTE_STATE_ROOT/snapshot_ready'" || return 1
  log 'Remote snapshot, baseline checkpoint, and batch scripts are ready'
}

audit_remote_script() {
  local script=$1 effective_time=$2 expected_minutes=$3 cpu_only=$4
  local args="--effective-walltime $effective_time"
  [ -n "$expected_minutes" ] && args="$args --expected-minutes $expected_minutes"
  [ "$cpu_only" = yes ] && args="$args --cpu-only"
  remote "/bin/bash -n '$REMOTE_SCRIPT_ROOT/$script' && /usr/bin/python3 '$AUDIT_SCRIPT' '$REMOTE_SCRIPT_ROOT/$script' $args" >> "$LOG_FILE" 2>&1
}

submit_stage() {
  local stage=$1 job_name=$2 script=$3 dependency=$4 exports=$5 effective_time=$6 expected_minutes=$7 cpu_only=$8
  local existing output jid
  [ -n "$(stage_jid "$stage")" ] && return 0
  if stage_failed "$stage"; then
    log "BLOCKED: $stage has a recorded failure; preserving deployed scripts"
    return 1
  fi
  existing=$(remote "/usr/bin/squeue -h -u mingyl -n '$job_name' -o '%A' | /usr/bin/head -1" 2>/dev/null || true)
  if [[ "$existing" =~ ^[0-9]+$ ]]; then
    remote_mark "$REMOTE_STATE_ROOT/$stage.jobid" "$existing"
    log "Attached $stage to existing job $existing"
    return 0
  fi
  queue_snapshot
  if ! audit_remote_script "$script" "$effective_time" "$expected_minutes" "$cpu_only"; then
    log "ERROR: resource audit/bash check failed for $stage; no submission made"
    return 1
  fi
  local command="sbatch --parsable --job-name=$job_name --time=$effective_time --export=ALL,$exports"
  [ -n "$dependency" ] && command="$command --dependency=afterok:$dependency"
  output=$(remote "$command '$REMOTE_SCRIPT_ROOT/$script'" 2>&1)
  if [[ "$output" =~ ^[0-9]+$ ]]; then
    jid=$output
    remote_mark "$REMOTE_STATE_ROOT/$stage.jobid" "$jid"
    log "Submitted $stage as job $jid dependency=${dependency:-none} walltime=$effective_time"
    return 0
  fi
  log "SUBMISSION FAILED for $stage: $output"
  if [[ "$output" == *AssocMaxSubmitJobLimit* || "$output" == *QOSMaxSubmitJobPerUserLimit* || "$output" == *MaxSubmitJob* ]]; then
    remote_mark "$REMOTE_STATE_ROOT/$stage.failed" quota_limit
    log "QUOTA BLOCK: preserving scripts and waiting for user direction"
  fi
  return 1
}

stage_finished() {
  local stage=$1 marker=$2 result_path=${3:-}
  local jid state log_path
  jid=$(stage_jid "$stage")
  [ -n "$jid" ] || return 1
  state=$(job_state "$jid")
  log "stage=$stage job=$jid state=${state:-UNKNOWN}"
  if state_is_failure "$state"; then
    log "FAILED: $stage job $jid state=$state"
    sync_stage_logs "$stage" "$jid"
    remote_mark "$REMOTE_STATE_ROOT/$stage.failed" "$state"
    return 2
  fi
  state_is_success "$state" || return 1
  log_path=$(stage_log_path "$stage" "$jid")
  if ! remote_file_contains "$log_path" "$marker"; then
    log "FAILED: $stage job $jid completed without marker $marker"
    sync_stage_logs "$stage" "$jid"
    remote_mark "$REMOTE_STATE_ROOT/$stage.failed" missing_marker
    return 2
  fi
  if [ -n "$result_path" ] && ! remote_has "$result_path"; then
    log "FAILED: $stage job $jid has no result $result_path"
    sync_stage_logs "$stage" "$jid"
    remote_mark "$REMOTE_STATE_ROOT/$stage.failed" missing_result
    return 2
  fi
  sync_stage_logs "$stage" "$jid"
  return 0
}

stage_ready() {
  local prereq status
  for prereq in "$@"; do
    case "$prereq" in
      smoke) stage_finished smoke ATHENA_LUMI_TRI_MEMORY_SMOKE_COMPLETE || return 1 ;;
      train) stage_finished train ATHENA_TRI_MEMORY_JOINT_TRAINING_COMPLETE "$(stage_result_path train)" || return 1 ;;
      eval_*) stage_finished "$prereq" "ATHENA_TRI_MEMORY_EVAL_COMPLETE ${prereq#eval_}" "$(stage_result_path "$prereq")" || return 1 ;;
    esac
  done
  return 0
}

process_stage() {
  local stage=$1 job_name=$2 script=$3 dependency=$4 exports=$5 effective_time=$6 expected_minutes=$7 cpu_only=$8
  local marker=${9:-} result=${10:-} jid status dep dep_jid dependency_ids
  jid=$(stage_jid "$stage")
  if [ -n "$jid" ]; then
    stage_finished "$stage" "$marker" "$result"
    return 0
  fi
  if stage_failed "$stage"; then
    log "stage=$stage remains blocked after prior failure"
    return 0
  fi
  dependency_ids=
  if [ -n "$dependency" ]; then
    for dep in $dependency; do
      if ! stage_ready "$dep"; then
        log "Waiting: prerequisites for $stage are not complete ($dep)"
        return 0
      fi
      dep_jid=$(stage_jid "$dep")
      if [ -z "$dep_jid" ]; then
        log "Waiting: prerequisite $dep has no recorded job ID"
        return 0
      fi
      if [ -n "$dependency_ids" ]; then dependency_ids="$dependency_ids:$dep_jid"; else dependency_ids=$dep_jid; fi
    done
  fi
  submit_stage "$stage" "$job_name" "$script" "$dependency_ids" "$exports" "$effective_time" "$expected_minutes" "$cpu_only" || true
}

write_final_report() {
  [ -f "$REPORT_FILE" ] && /usr/bin/grep -F -q ATHENA_TRI_MEMORY_PIPELINE_COMPLETE "$REPORT_FILE" && return 0
  local tmp=$REPORT_FILE
  {
    /usr/bin/printf '# ATHENA tri-memory pipeline report\n\n'
    /usr/bin/printf -- '- completed_at_utc: %s\n' "$(timestamp)"
    /usr/bin/printf -- '- pipeline_tag: %s\n' "$PIPELINE_TAG"
    /usr/bin/printf -- '- remote_root: %s\n' "$REMOTE_ROOT"
    /usr/bin/printf -- '- local_job_logs: %s\n\n' "$LOCAL_JOB_LOG_ROOT"
    for stage in smoke train eval_nq eval_webqa eval_triviaqa eval_truthfulqa eval_hotpotqa bootstrap oracle; do
      local jid state
      jid=$(stage_jid "$stage")
      state=NOT_SUBMITTED
      [ -n "$jid" ] && state=$(job_state "$jid")
      /usr/bin/printf -- '- %s: job=%s state=%s\n' "$stage" "${jid:-none}" "$state"
    done
    /usr/bin/printf '\nATHENA_TRI_MEMORY_PIPELINE_COMPLETE\n'
  } > "$tmp"
  log "Pipeline complete; report written to $REPORT_FILE"
}

main() {
  log 'Monitor cycle started'
  queue_snapshot
  ensure_remote_layout || { log 'ERROR: remote layout preparation failed'; return 1; }

  process_stage smoke athena-tri-softsubset-smoke lumi_tri_memory_smoke.slurm \
    '' "EXPERIMENT_TAG=$PIPELINE_TAG" 00:05:00 5 no \
    ATHENA_LUMI_TRI_MEMORY_SMOKE_COMPLETE ''
  process_stage train athena-tri-softsubset20m lumi_tri_memory_joint.slurm \
    smoke "EXPERIMENT_TAG=$PIPELINE_TAG" 1-16:00:00 2100 no \
    ATHENA_TRI_MEMORY_JOINT_TRAINING_COMPLETE "$(stage_result_path train)"

  local task stage
  for task in nq webqa triviaqa truthfulqa hotpotqa; do
    case "$task" in
      nq) effective=12:00:00; expected=480 ;;
      webqa) effective=10:00:00; expected=360 ;;
      triviaqa) effective=2-00:00:00; expected=2280 ;;
      truthfulqa) effective=16:00:00; expected=600 ;;
      hotpotqa) effective=1-06:00:00; expected=1080 ;;
    esac
    stage=eval_$task
    process_stage "$stage" "athena-tri-softsubset-eval-$task" lumi_tri_memory_eval.slurm \
      train "EXPERIMENT_TAG=$PIPELINE_TAG,TASK=$task" "$effective" "$expected" no \
      "ATHENA_TRI_MEMORY_EVAL_COMPLETE $task" "$(stage_result_path "$stage")"
  done

  local eval_dependency='eval_nq eval_webqa eval_triviaqa eval_truthfulqa eval_hotpotqa'
  process_stage bootstrap athena-tri-softsubset-bootstrap lumi_tri_memory_bootstrap.slurm \
    "$eval_dependency" "EXPERIMENT_TAG=$PIPELINE_TAG" 00:30:00 15 yes \
    ATHENA_TRI_MEMORY_BOOTSTRAP_COMPLETE "$(stage_result_path bootstrap)"
  process_stage oracle athena-tri-softsubset-oracle lumi_tri_memory_corrected_oracle.slurm \
    "$eval_dependency" "EXPERIMENT_TAG=$PIPELINE_TAG" 00:30:00 15 yes \
    ATHENA_TRI_MEMORY_CORRECTED_ORACLE_COMPLETE "$(stage_result_path oracle)"

  if stage_finished bootstrap ATHENA_TRI_MEMORY_BOOTSTRAP_COMPLETE "$(stage_result_path bootstrap)" \
    && stage_finished oracle ATHENA_TRI_MEMORY_CORRECTED_ORACLE_COMPLETE "$(stage_result_path oracle)"; then
    write_final_report
  fi
  log 'Monitor cycle finished'
}

main "$@"
