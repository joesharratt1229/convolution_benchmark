
sudo ncu -f --set full --export report.ncu-rep ./conv2b
sudo ncu -f --set detailed --export report.ncu-rep ./conv2b
sudo ncu --section SpeedOfLight --kernel-name pos_embedding_kernel --launch-skip 1 --launch-count 1 ./conv2b
sudo ncu -c 3 --section SpeedOfLight ./conv2b 