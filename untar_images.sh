cd /mnt/hdfs/andrew.estornell/vlm/MMDS-SampledDataPool

cat images.tar.zst.* | tar --zstd -xvf -

cd -