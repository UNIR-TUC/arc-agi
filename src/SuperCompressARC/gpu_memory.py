def capture(cuda):
    """Capture process-local allocator peaks and current device-wide usage."""
    cuda.synchronize()
    free_bytes, total_bytes = cuda.mem_get_info()
    return {
        'peak_allocated_bytes': int(cuda.max_memory_allocated()),
        'peak_reserved_bytes': int(cuda.max_memory_reserved()),
        'device_used_bytes': int(total_bytes - free_bytes),
    }


def summarize_run_metadata(run_metadata, phase='phase2_workers'):
    """Aggregate worker GPU reports while tolerating legacy run metadata."""
    reports = []
    for metadata in run_metadata:
        phase_reports = (metadata.get('gpu_memory') or {}).get(phase) or {}
        if isinstance(phase_reports, dict):
            reports.extend(report for report in phase_reports.values()
                           if isinstance(report, dict))

    def maximum(field):
        values = [report[field] for report in reports
                  if isinstance(report.get(field), (int, float))]
        return max(values) if values else None

    return {
        'worker_reports': len(reports),
        'process_peak_allocated_max_bytes': maximum('peak_allocated_bytes'),
        'process_peak_reserved_max_bytes': maximum('peak_reserved_bytes'),
        'device_used_at_worker_exit_max_bytes': maximum('device_used_bytes'),
    }