#!/bin/bash
distrobox create \
--hostname grasping \
-n grasping \
-i danielmunicio/omnidrones:eecs \
--volume /home:/home:rslave \
--volume /share:/share:rslave \
--volume /scratch:/scratch \
--volume /var/lib/sss/pipes:/var/lib/sss/pipes \
-a "--device nvidia.com/gpu=all" \
--pre-init-hooks "sudo mount -t tmpfs -o mode=1777,nodev,nosuid tmpfs /tmp"
