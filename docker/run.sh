docker run -it \
 -v /home/albert/repos/albert/swebench/swe-bench/:/shared \
 -v /home/albert/miniconda3/envs:/home/swe-bench/miniconda3/envs \
 opendevin/swe-bench:latest bash
