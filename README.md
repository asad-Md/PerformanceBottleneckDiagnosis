# PerformanceBottleneckDiagnosis
AI Assisted Performance Bottleneck Diagnosis for Multicore Systems

clang -O2 -g -target bpf -D__TARGET_ARCH_x86 -I/usr/include/bpf -c perf_monitor.bpf.c -o perf_monitor.bpf.o

gcc -O2 -o pinner pinner.c -lbpf -lelf -lz
gcc -O2 -o reader reader.c -lbpf -lelf -lz

sudo ./pinner --obj perf_monitor.bpf.o
sudo ./reader --label cpu_bound --skip-start 5 --skip-end 2



AUTODATA.SH 

Create the script:

* Open a file named `autoData.sh`
* Paste the script content into it
* Edit the `PASSWORD` field (or remove it if using sudo cache)

Save the file:

* Save and exit the editor

Make it executable:

* Run: `chmod +x autoData.sh`

(Optional) cache sudo credentials:

* Run: `sudo -v`

Run the script:

* Execute: `./autoData.sh`

Monitor execution:

* Watch terminal output for each session label
* Ensure stress-ng, pinner, and reader run in sequence

Verify output:

* Check that `perf_metrics.csv` is being appended
* Confirm entries for each session label are present
