#!/usr/bin/env bash

# Idempotent LUMI pipeline monitor.  It is intentionally a short-lived
# --once process; launchd invokes it every 30 minutes.  No scancel operation is
# present by design.

set -uo pipefail

PROJECT_ROOT=/Users/mingyl/ATHENA
REMOTE_HOST=lumi.csc.fi
REMOTE_ROOT=/scratch/project_462001493/mingyl/ATHENA
REMOTE_CACHE_ROOT=/scratch/project_462001050
PIPELINE_TAG=tri_memory_joint_20260902T180738Z
REMOTE_SNAPSHOT=$REMOTE_ROOT/snapshots/$PIPELINE_TAG
REMOTE_SCRIPT_ROOT=$REMOTE_ROOT/scripts/$PIPELINE_TAG
REMOTE_RUN_ROOT=$REMOTE_ROOT/run/$PIPELINE_TAG
REMOTE_STATE_ROOT=$REMOTE_RUN_ROOT/state
OLD_RUN_ROOT=$REMOTE_ROOT/run/wiki_fair_joint_20260901T173940Z
LOCAL_LOG_ROOT=$PROJECT_ROOT/logs/csc/lumi/scheduler
LOCAL_JOB_LOG_ROOT=$PROJECT_ROOT/logs/csc/lumi
LOG_FILE=$LOCAL_LOG_ROOT/monitor.log
REPORT_FILE=$LOCAL_LOG_ROOT/pipeline_report.md
LOCK_DIR=$LOCAL_LOG_ROOT/.lock
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
  /usr/bin/ssh \
    -o BatchMode=yes \
    -o ConnectTimeout=20 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=2 \
    "$REMOTE_HOST" "$@"
}

remote_value() {
  local path=$1
  remote "if test -f '$path'; then /bin/cat '$path'; fi" 2>/dev/null \
    | /usr/bin/tr -d '\r\n '
}

remote_mark() {
  local path=$1
  local value=$2
  # printf is a POSIX shell builtin on the cluster; /bin/printf is not
  # available on the current LUMI login environment.
  remote "printf '%s\\n' '$value' > '$path'" >/dev/null
}

remote_has() {
  local path=$1
  remote "test -e '$path'"
}

remote_file_contains() {
  local path=$1
  local needle=$2
  remote "/usr/bin/grep -F -q -- '$needle' '$path'"
}

queue_snapshot() {
  log "squeue before submission:"
  remote "/usr/bin/squeue -u mingyl -o '%.18i %.12P %.32j %.10T %.12M %.12l %R'" \
    >> "$LOG_FILE" 2>&1 || log "WARNING: unable to read squeue"
}

job_state() {
  local jid=$1
  local state
  state=$(remote "/usr/bin/sacct -X -n -P -j '$jid' --format=State | /usr/bin/head -1 | /usr/bin/tr -d '[:space:]'" 2>/dev/null || true)
  if [ -z "$state" ]; then
    state=$(remote "/usr/bin/squeue -h -j '$jid' -o '%T' | /usr/bin/head -1 | /usr/bin/tr -d '[:space:]'" 2>/dev/null || true)
  fi
  /usr/bin/printf '%s' "$state"
}

state_is_success() {
  case "$1" in
    COMPLETED) return 0 ;;
    *) return 1 ;;
  esac
}

state_is_failure() {
  case "$1" in
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED) return 0 ;;
    *) return 1 ;;
  esac
}

state_is_active() {
  case "$1" in
    PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING) return 0 ;;
    *) return 1 ;;
  esac
}

stage_jid() {
  remote_value "$REMOTE_STATE_ROOT/$1.jobid"
}

stage_blocked() {
  remote_has "$REMOTE_STATE_ROOT/$1.blocked"
}

stage_status() {
  local stage=$1
  local jid
  jid=$(stage_jid "$stage")
  if [ -z "$jid" ]; then
    /usr/bin/printf 'NOT_SUBMITTED'
  else
    job_state "$jid"
  fi
}

stage_log_path() {
  local stage=$1
  local jid=$2
  case "$stage" in
    smoke) /usr/bin/printf '%s/logs/tri-smoke-%s.out' "$REMOTE_ROOT" "$jid" ;;
    correction) /usr/bin/printf '%s/logs/ge-correction-%s.out' "$REMOTE_ROOT" "$jid" ;;
    correction_bootstrap) /usr/bin/printf '%s/logs/ge-correction-stats-%s.out' "$REMOTE_ROOT" "$jid" ;;
    baseline) /usr/bin/printf '%s/logs/tri-baseline-%s.out' "$REMOTE_ROOT" "$jid" ;;
    tri_train) /usr/bin/printf '%s/logs/tri-memory-train-%s.out' "$REMOTE_ROOT" "$jid" ;;
    eval_nq|eval_webqa|eval_triviaqa|eval_truthfulqa|eval_hotpotqa)
      /usr/bin/printf '%s/logs/tri-eval-%s.out' "$REMOTE_ROOT" "$jid"
      ;;
    tri_bootstrap) /usr/bin/printf '%s/logs/tri-memory-stats-%s.out' "$REMOTE_ROOT" "$jid" ;;
    corrected_oracle) /usr/bin/printf '%s/logs/tri-oracle-fix-%s.out' "$REMOTE_ROOT" "$jid" ;;
    *) return 1 ;;
  esac
}

# The current tri-memory experiment was submitted before this monitor was
# enabled.  Discover and record those jobs without submitting replacements.
# The job-name lookup keeps this monitor useful after a queue restart; the
# fixed IDs are a one-time fallback for jobs that finish between two cycles.
current_stage_job_name() {
  case "$1" in
    baseline) /usr/bin/printf '%s' athena-tri-baseline20m ;;
    correction) /usr/bin/printf '%s' athena-ge-correction ;;
    correction_bootstrap) /usr/bin/printf '%s' athena-ge-correction-stats ;;
    tri_train) /usr/bin/printf '%s' athena-tri-memory20m ;;
    eval_nq) /usr/bin/printf '%s' athena-tri-eval-nq ;;
    eval_webqa) /usr/bin/printf '%s' athena-tri-eval-webqa ;;
    eval_triviaqa) /usr/bin/printf '%s' athena-tri-eval-triviaqa ;;
    eval_truthfulqa) /usr/bin/printf '%s' athena-tri-eval-truthfulqa ;;
    eval_hotpotqa) /usr/bin/printf '%s' athena-tri-eval-hotpotqa ;;
    tri_bootstrap) /usr/bin/printf '%s' athena-tri-memory-stats ;;
    corrected_oracle) /usr/bin/printf '%s' athena-tri-oracle-fix ;;
    *) return 1 ;;
  esac
}

current_stage_fallback_id() {
  case "$1" in
    baseline) /usr/bin/printf '%s' 21684275 ;;
    correction) /usr/bin/printf '%s' 21684745 ;;
    correction_bootstrap) /usr/bin/printf '%s' 21684281 ;;
    tri_train) /usr/bin/printf '%s' 21684282 ;;
    eval_nq) /usr/bin/printf '%s' 21684283 ;;
    eval_webqa) /usr/bin/printf '%s' 21684285 ;;
    eval_triviaqa) /usr/bin/printf '%s' 21684286 ;;
    eval_truthfulqa) /usr/bin/printf '%s' 21684287 ;;
    eval_hotpotqa) /usr/bin/printf '%s' 21684288 ;;
    tri_bootstrap) /usr/bin/printf '%s' 21684289 ;;
    corrected_oracle) /usr/bin/printf '%s' 21686012 ;;
    *) return 1 ;;
  esac
}

seed_current_pipeline_state() {
  local stage job_name jid fallback
  for stage in baseline correction correction_bootstrap tri_train \
    eval_nq eval_webqa eval_triviaqa eval_truthfulqa eval_hotpotqa \
    tri_bootstrap corrected_oracle; do
    if [ -n "$(stage_jid "$stage")" ]; then
      continue
    fi
    job_name=$(current_stage_job_name "$stage")
    jid=$(remote "/usr/bin/squeue -h -u mingyl -n '$job_name' -o '%A' | /usr/bin/head -1" 2>/dev/null || true)
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
      fallback=$(current_stage_fallback_id "$stage")
      if [ -n "$(job_state "$fallback")" ]; then
        jid=$fallback
      fi
    fi
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
      remote_mark "$REMOTE_STATE_ROOT/$stage.jobid" "$jid"
      log "Attached current $stage to existing job $jid ($job_name)"
    else
      log "WARNING: current $stage job was not found ($job_name)"
    fi
  done
}

sync_current_baseline_log() {
  local jid state destination
  jid=$(stage_jid baseline)
  [ -n "$jid" ] || return 0
  state=$(job_state "$jid")
  if state_is_success "$state" || state_is_failure "$state"; then
    destination=$LOCAL_JOB_LOG_ROOT/$jid
    /bin/mkdir -p "$destination"
    copy_remote_file "$(stage_log_path baseline "$jid")" "$destination/stdout.out"
    copy_remote_file "$(stage_err_path baseline "$jid")" "$destination/stderr.err"
  fi
}

stage_err_path() {
  local stage=$1
  local jid=$2
  local out
  out=$(stage_log_path "$stage" "$jid") || return 1
  /usr/bin/printf '%s' "${out%.out}.err"
}

copy_remote_file() {
  local remote_path=$1
  local local_path=$2
  if remote_has "$remote_path"; then
    /bin/mkdir -p "$(/usr/bin/dirname "$local_path")"
    /usr/bin/scp \
      -q \
      -o BatchMode=yes \
      -o ConnectTimeout=20 \
      "$REMOTE_HOST:$remote_path" "$local_path" \
      >> "$LOG_FILE" 2>&1 || log "WARNING: failed to copy $remote_path"
  fi
}

sync_stage_logs() {
  local stage=$1
  local jid=$2
  local destination=$LOCAL_JOB_LOG_ROOT/$jid
  if [ -f "$destination/.downloaded" ]; then
    return 0
  fi
  /bin/mkdir -p "$destination"
  copy_remote_file "$(stage_log_path "$stage" "$jid")" "$destination/stdout.out"
  copy_remote_file "$(stage_err_path "$stage" "$jid")" "$destination/stderr.err"
  case "$stage" in
    baseline)
      copy_remote_file \
        "$REMOTE_RUN_ROOT/engram_20m_train/results.json" \
        "$destination/results.json"
      copy_remote_file \
        "$REMOTE_RUN_ROOT/engram_20m_train/config.json" \
        "$destination/config.json"
      ;;
    correction)
      copy_remote_file \
        "$OLD_RUN_ROOT/joint_generated_only_correction_nq/results.json" \
        "$destination/results.json"
      ;;
    correction_bootstrap)
      copy_remote_file \
        "$OLD_RUN_ROOT/joint_generated_only_correction_nq/results_with_bootstrap.json" \
        "$destination/results_with_bootstrap.json"
      ;;
    tri_train)
      copy_remote_file "$REMOTE_RUN_ROOT/tri_20m_train/results.json" "$destination/results.json"
      copy_remote_file "$REMOTE_RUN_ROOT/tri_20m_train/config.json" "$destination/config.json"
      ;;
    eval_nq|eval_webqa|eval_triviaqa|eval_truthfulqa|eval_hotpotqa)
      local task=${stage#eval_}
      copy_remote_file "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results.json" "$destination/results.json"
      ;;
    tri_bootstrap)
      local task
      for task in nq webqa triviaqa truthfulqa hotpotqa; do
        copy_remote_file \
          "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results_with_bootstrap.json" \
          "$destination/${task}_results_with_bootstrap.json"
      done
      ;;
    corrected_oracle)
      local task
      for task in nq webqa triviaqa truthfulqa hotpotqa; do
        copy_remote_file \
          "$REMOTE_RUN_ROOT/${task}_tri_full_eval/results_with_corrected_oracle.json" \
          "$destination/${task}_results_with_corrected_oracle.json"
      done
      ;;
  esac
  /usr/bin/touch "$destination/.downloaded"
}

sync_old_logs() {
  local jid stage out err destination
  local -a jobs=(21675599 21675600 21675601 21675602 21675603 21675604 21675605 21675606)
  local old_log_root=$REMOTE_ROOT/logs/wiki_fair_joint_20260901T173940Z
  for jid in "${jobs[@]}"; do
    case "$jid" in
      21675599) out="$old_log_root/train-athena-fair-engram20m-$jid.out" ;;
      21675600) out="$old_log_root/train-athena-fair-joint20m-$jid.out" ;;
      21675601) out="$old_log_root/eval-athena-fair-nq-$jid.out" ;;
      21675602) out="$old_log_root/eval-athena-fair-webqa-$jid.out" ;;
      21675603) out="$old_log_root/eval-athena-fair-trivia-$jid.out" ;;
      21675604) out="$old_log_root/eval-athena-fair-truth-$jid.out" ;;
      21675605) out="$old_log_root/eval-athena-fair-hotpot-$jid.out" ;;
      21675606) out="$old_log_root/stats-$jid.out" ;;
    esac
    err="${out%.out}.err"
    destination=$LOCAL_JOB_LOG_ROOT/$jid
    if [ -f "$destination/.downloaded" ]; then
      continue
    fi
    local old_state
    old_state=$(job_state "$jid")
    if ! state_is_success "$old_state" && ! state_is_failure "$old_state"; then
      continue
    fi
    /bin/mkdir -p "$destination"
    copy_remote_file "$out" "$destination/stdout.out"
    copy_remote_file "$err" "$destination/stderr.err"
    /usr/bin/touch "$destination/.downloaded"
  done
}

ensure_remote_layout() {
  if remote_has "$REMOTE_STATE_ROOT/snapshot_ready"; then
    return 0
  fi
  log "Preparing isolated remote snapshot $PIPELINE_TAG"
  if ! remote "/bin/mkdir -p '$REMOTE_SNAPSHOT' '$REMOTE_SCRIPT_ROOT' '$REMOTE_RUN_ROOT' '$REMOTE_RUN_ROOT/logs' '$REMOTE_STATE_ROOT' '$REMOTE_ROOT/logs'"; then
    log "ERROR: unable to create remote pipeline directories"
    return 1
  fi
  if ! /usr/bin/rsync -a \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    "$PROJECT_ROOT/engram/" "$REMOTE_HOST:$REMOTE_SNAPSHOT/engram/" \
    >> "$LOG_FILE" 2>&1; then
    log "ERROR: failed to sync engram source"
    return 1
  fi
  if ! /usr/bin/rsync -a \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    "$PROJECT_ROOT/scripts/" "$REMOTE_HOST:$REMOTE_SNAPSHOT/scripts/" \
    >> "$LOG_FILE" 2>&1; then
    log "ERROR: failed to sync scripts source"
    return 1
  fi
  local script
  for script in \
    lumi_tri_memory_smoke.slurm \
    lumi_generated_only_correction.slurm \
    lumi_generated_only_bootstrap.slurm \
    lumi_tri_memory_joint.slurm \
    lumi_tri_memory_eval.slurm \
    lumi_tri_memory_bootstrap.slurm; do
    if ! /usr/bin/rsync -a "$PROJECT_ROOT/run/$script" "$REMOTE_HOST:$REMOTE_SCRIPT_ROOT/$script" \
      >> "$LOG_FILE" 2>&1; then
      log "ERROR: failed to sync $script"
      return 1
    fi
  done
  if ! remote "/bin/chmod +x '$REMOTE_SCRIPT_ROOT'/*.slurm && /bin/touch '$REMOTE_STATE_ROOT/snapshot_ready'"; then
    log "ERROR: failed to finalize remote snapshot"
    return 1
  fi
  log "Remote snapshot and batch scripts are ready"
}

audit_remote_script() {
  local script=$1
  local effective_time=$2
  local expected_minutes=$3
  local cpu_only=$4
  local audit_args="--effective-walltime $effective_time"
  if [ -n "$expected_minutes" ]; then
    audit_args="$audit_args --expected-minutes $expected_minutes"
  fi
  if [ "$cpu_only" = yes ]; then
    audit_args="$audit_args --cpu-only"
  fi
  remote "/bin/bash -n '$REMOTE_SCRIPT_ROOT/$script' && /usr/bin/python3 '$AUDIT_SCRIPT' '$REMOTE_SCRIPT_ROOT/$script' $audit_args" \
    >> "$LOG_FILE" 2>&1
}

submit_stage() {
  local stage=$1
  local job_name=$2
  local script=$3
  local dependency=$4
  local exports=$5
  local effective_time=$6
  local expected_minutes=$7
  local cpu_only=$8
  local jid existing submit_output

  if [ -n "$(stage_jid "$stage")" ]; then
    return 0
  fi
  if stage_blocked "$stage"; then
    log "BLOCKED: $stage has a prior quota/submission block; leaving script deployed"
    return 1
  fi

  existing=$(remote "/usr/bin/squeue -h -u mingyl -n '$job_name' -o '%A' | /usr/bin/head -1" 2>/dev/null || true)
  if [[ "$existing" =~ ^[0-9]+$ ]]; then
    remote_mark "$REMOTE_STATE_ROOT/$stage.jobid" "$existing"
    log "Attached $stage to existing job $existing ($job_name)"
    return 0
  fi

  queue_snapshot
  if ! audit_remote_script "$script" "$effective_time" "$expected_minutes" "$cpu_only"; then
    log "ERROR: resource audit or bash -n failed for $stage; no submission made"
    return 1
  fi

  local command="sbatch --parsable --job-name=$job_name --time=$effective_time --export=ALL,$exports"
  if [ -n "$dependency" ]; then
    command="$command --dependency=afterok:$dependency"
  fi
  command="$command $REMOTE_SCRIPT_ROOT/$script"
  submit_output=$(remote "$command" 2>&1)
  if [[ "$submit_output" =~ ^[0-9]+$ ]]; then
    jid=$submit_output
    remote_mark "$REMOTE_STATE_ROOT/$stage.jobid" "$jid"
    log "Submitted $stage as job $jid ($job_name), dependency=${dependency:-none}, reservation=${effective_time}"
    return 0
  fi

  log "SUBMISSION FAILED for $stage: $submit_output"
  if [[ "$submit_output" == *AssocMaxSubmitJobLimit* || "$submit_output" == *QOSMaxSubmitJobPerUserLimit* || "$submit_output" == *MaxSubmitJob* ]]; then
    remote_mark "$REMOTE_STATE_ROOT/$stage.blocked" "quota_limit"
    log "QUOTA BLOCK: preserving $script and waiting for explicit user direction"
  fi
  return 1
}

stage_finished() {
  local stage=$1
  local marker=$2
  local result_path=${3:-}
  local jid state log_path
  jid=$(stage_jid "$stage")
  [ -n "$jid" ] || return 1
  state=$(job_state "$jid")
  log_path=$(stage_log_path "$stage" "$jid")
  if state_is_failure "$state"; then
    log "FAILED: $stage job $jid state=$state"
    sync_stage_logs "$stage" "$jid"
    return 2
  fi
  if ! state_is_success "$state"; then
    return 1
  fi
  if ! remote_file_contains "$log_path" "$marker"; then
    log "FAILED: $stage job $jid completed without marker $marker"
    sync_stage_logs "$stage" "$jid"
    return 2
  fi
  if [ -n "$result_path" ] && ! remote_has "$result_path"; then
    log "FAILED: $stage job $jid has no expected result $result_path"
    sync_stage_logs "$stage" "$jid"
    return 2
  fi
  sync_stage_logs "$stage" "$jid"
  return 0
}

old_training_ready() {
  local jid state
  for jid in 21675599 21675600; do
    state=$(job_state "$jid")
    if state_is_failure "$state"; then
      log "BLOCKED: preserved old training job $jid ended in $state"
      return 2
    fi
    if ! state_is_success "$state"; then
      return 1
    fi
  done
  if ! remote_has "$OLD_RUN_ROOT/engram_20m_train/adaptor_best.pt" || \
     ! remote_has "$OLD_RUN_ROOT/joint_20m_train/adaptor_best.pt"; then
    return 1
  fi
  return 0
}

all_tri_evals_finished() {
  local task stage result marker status
  for task in nq webqa triviaqa truthfulqa hotpotqa; do
    stage=eval_$task
    result_path="$REMOTE_RUN_ROOT/${task}_tri_full_eval/results.json"
    marker="ATHENA_TRI_MEMORY_EVAL_COMPLETE $task"
    stage_finished "$stage" "$marker" "$result_path"
    status=$?
    if [ "$status" -eq 2 ]; then
      return 2
    fi
    if [ "$status" -ne 0 ]; then
      return 1
    fi
  done
  return 0
}

write_final_report() {
  if [ -f "$REPORT_FILE" ] && /usr/bin/grep -F -q 'ATHENA_PIPELINE_COMPLETE' "$REPORT_FILE"; then
    return 0
  fi
  local temporary=$REPORT_FILE.tmp
  {
    /usr/bin/printf '# ATHENA LUMI pipeline report\n\n'
    /usr/bin/printf -- '- completed_at_utc: %s\n' "$(timestamp)"
    /usr/bin/printf -- '- pipeline_tag: %s\n' "$PIPELINE_TAG"
    /usr/bin/printf -- '- remote_root: %s\n' "$REMOTE_ROOT"
    /usr/bin/printf -- '- local_job_logs: %s\n\n' "$LOCAL_JOB_LOG_ROOT"
    /usr/bin/printf '## Managed jobs\n\n'
    local stage jid state
    for stage in baseline correction correction_bootstrap tri_train eval_nq eval_webqa eval_triviaqa eval_truthfulqa eval_hotpotqa tri_bootstrap corrected_oracle; do
      jid=$(stage_jid "$stage")
      state=NOT_SUBMITTED
      if [ -n "$jid" ]; then state=$(job_state "$jid"); fi
      /usr/bin/printf -- '- %s: job=%s state=%s\n' "$stage" "${jid:-none}" "$state"
    done
    /usr/bin/printf '\n## Existing preserved jobs\n\n'
    for jid in 21675599 21675600 21675601 21675602 21675603 21675604 21675605 21675606; do
      /usr/bin/printf -- '- job=%s state=%s\n' "$jid" "$(job_state "$jid")"
    done
    /usr/bin/printf '\nATHENA_PIPELINE_COMPLETE\n'
  } > "$temporary" && /bin/mv "$temporary" "$REPORT_FILE"
  log "Pipeline complete; report written to $REPORT_FILE"
}

monitor_existing_pipeline() {
  local stage jid state status
  seed_current_pipeline_state
  sync_current_baseline_log

  for stage in baseline correction correction_bootstrap tri_train \
    eval_nq eval_webqa eval_triviaqa eval_truthfulqa eval_hotpotqa \
    tri_bootstrap corrected_oracle; do
    jid=$(stage_jid "$stage")
    if [ -z "$jid" ]; then
      log "Waiting: no job ID recorded for $stage"
      continue
    fi
    state=$(job_state "$jid")
    log "stage=$stage job=$jid state=${state:-UNKNOWN}"
    if state_is_failure "$state"; then
      sync_stage_logs "$stage" "$jid"
      log "ERROR: current pipeline stage failed: $stage job $jid state=$state"
      return 1
    fi
    if ! state_is_success "$state"; then
      continue
    fi

    case "$stage" in
      baseline)
        stage_finished baseline ATHENA_TRI_MEMORY_MATCHED_BASELINE_COMPLETE \
          "$REMOTE_RUN_ROOT/engram_20m_train/results.json"
        ;;
      correction)
        stage_finished correction ATHENA_GE_CORRECTION_EVAL_COMPLETE \
          "$OLD_RUN_ROOT/joint_generated_only_correction_nq/results.json"
        ;;
      correction_bootstrap)
        stage_finished correction_bootstrap ATHENA_GE_CORRECTION_BOOTSTRAP_COMPLETE \
          "$OLD_RUN_ROOT/joint_generated_only_correction_nq/results_with_bootstrap.json"
        ;;
      tri_train)
        stage_finished tri_train ATHENA_TRI_MEMORY_JOINT_TRAINING_COMPLETE \
          "$REMOTE_RUN_ROOT/tri_20m_train/results.json"
        ;;
      eval_nq|eval_webqa|eval_triviaqa|eval_truthfulqa|eval_hotpotqa)
        stage_finished "$stage" "ATHENA_TRI_MEMORY_EVAL_COMPLETE ${stage#eval_}" \
          "$REMOTE_RUN_ROOT/${stage#eval_}_tri_full_eval/results.json"
        ;;
      tri_bootstrap)
        stage_finished tri_bootstrap ATHENA_TRI_MEMORY_BOOTSTRAP_COMPLETE \
          "$REMOTE_RUN_ROOT/nq_tri_full_eval/results_with_bootstrap.json"
        ;;
      corrected_oracle)
        stage_finished corrected_oracle ATHENA_TRI_MEMORY_CORRECTED_ORACLE_COMPLETE \
          "$REMOTE_RUN_ROOT/nq_tri_full_eval/results_with_corrected_oracle.json"
        ;;
    esac
    status=$?
    if [ "$status" -eq 2 ]; then
      return 1
    fi
  done

  local bootstrap_jid oracle_jid
  bootstrap_jid=$(stage_jid tri_bootstrap)
  oracle_jid=$(stage_jid corrected_oracle)
  if [ -n "$bootstrap_jid" ] && [ -n "$oracle_jid" ] \
    && state_is_success "$(job_state "$bootstrap_jid")" \
    && state_is_success "$(job_state "$oracle_jid")"; then
    if remote_has "$REMOTE_RUN_ROOT/nq_tri_full_eval/results_with_bootstrap.json" \
      && remote_has "$REMOTE_RUN_ROOT/nq_tri_full_eval/results_with_corrected_oracle.json"; then
      write_final_report
    fi
  fi
  log "Monitor cycle finished; no submissions made"
  return 0
}

main() {
  log "Monitor cycle started"
  queue_snapshot
  sync_old_logs
  if ! ensure_remote_layout; then
    return 1
  fi
  monitor_existing_pipeline
}

main "$@"
