import subprocess
import time
import psutil
import csv
import sys
import os

def main():
    if len(sys.argv) > 1:
        label = sys.argv[1]
    else:
        label = "benchmark"

    # Command to run complete demo with 20 tasks
    cmd = [sys.executable, "parallel_train.py", "--demo", "20"]
    
    print(f"Running benchmark for: {label}...")
    
    start_time = time.time()
    
    # Run the process. Inherit stdout/stderr directly so it prints the same as parallel_train.py
    proc = subprocess.Popen(cmd)
    
    try:
        ps_proc = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        print("Process failed to start.")
        return

    # Maintain a dictionary of pid -> psutil.Process object to correctly track cpu_percent
    processes = {ps_proc.pid: ps_proc}
    ps_proc.cpu_percent(interval=None)
    
    data_points = []
    
    try:
        while proc.poll() is None:
            time.sleep(1.0)
            
            try:
                # Update process list
                children = ps_proc.children(recursive=True)
                current_procs = [ps_proc] + children
                
                # Add new processes to our tracker
                for p in current_procs:
                    if p.pid not in processes:
                        processes[p.pid] = p
                        try:
                            p.cpu_percent(interval=None) # Initialize
                        except psutil.NoSuchProcess:
                            pass
                
                total_cpu_percent = 0.0
                total_cpu_time = 0.0
                
                for p in current_procs:
                    try:
                        total_cpu_percent += p.cpu_percent(interval=None)
                        times = p.cpu_times()
                        total_cpu_time += times.user + times.system
                    except psutil.NoSuchProcess:
                        pass
                
                current_time = time.time() - start_time
                data_points.append({
                    'time_s': round(current_time, 2),
                    'cpu_percent': round(total_cpu_percent, 2),
                    'cpu_time_s': round(total_cpu_time, 2)
                })
                
            except psutil.NoSuchProcess:
                break
                
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user. Terminating process...")
        try:
            ps_proc.terminate()
            ps_proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            proc.kill()
        # Proceed to save partial results
        
    proc.wait()
    end_time = time.time()
    
    if proc.returncode != 0:
        print(f"Warning: Process exited with non-zero code {proc.returncode}")
        
    wall_time = end_time - start_time
    
    # Save to CSV
    csv_filename = f"cpu_usage_{label}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['time_s', 'cpu_percent', 'cpu_time_s'])
        writer.writeheader()
        writer.writerows(data_points)
        
    # Calculate stats
    if data_points:
        avg_cpu_percent = sum(d['cpu_percent'] for d in data_points) / len(data_points)
        max_cpu_percent = max(d['cpu_percent'] for d in data_points)
        final_cpu_time = data_points[-1]['cpu_time_s']
    else:
        avg_cpu_percent = 0
        max_cpu_percent = 0
        final_cpu_time = 0
        
    # Save to TXT
    txt_filename = f"cpu_summary_{label}.txt"
    with open(txt_filename, 'w') as f:
        f.write(f"==============================================================\n")
        f.write(f"  Performance Summary — label: {label}\n")
        f.write(f"==============================================================\n")
        f.write(f"  Wall time        : {wall_time:.2f} s\n")
        f.write(f"  Total CPU time   : {final_cpu_time:.2f} s\n")
        f.write(f"  Average CPU %    : {avg_cpu_percent:.2f} %\n")
        f.write(f"  Max CPU %        : {max_cpu_percent:.2f} %\n")
        
    print(f"\nBenchmark finished.")
    print(f"Results saved to {csv_filename} and {txt_filename}.")

if __name__ == "__main__":
    main()
