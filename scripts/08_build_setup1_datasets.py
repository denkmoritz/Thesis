"""Setup 1 with human DVS zones: 3-channel images, boxes A, channel variants, label-free boxes B2.
Old steps 15, 15c, 16, 16b. PALMA: m3.slurm, m3_channels.slurm, m3_b2.slurm."""
from dti.datasets import setup1, channels, b2_box_rule, b2_write

setup1.main()
channels.main()
b2_box_rule.main()
b2_write.main()
