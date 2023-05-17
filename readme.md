bag下载地址

链接：https://pan.baidu.com/s/1DE8PUkGEX55W_lFk68A1Vg?pwd=slam 
提取码：slam

#### 1、建图

生成特征地图flat_cloud_map.pcd和sharp_cloud_map.pcd，关键帧SCDs文件夹，关键帧位姿 pose.txt，保存至lio_sam的data文件夹下

```
roslaunch lio_sam mapping.launch
rosbag play park.bag
```

然后，将data复制到data_park文件夹下，(建议换文件夹复制一下，防止被覆盖)

```
cp -r data data_park
```

#### 2、基于已有地图重定位

基于地图的定位和重定位（rviz手动重定位和sc自动重定位）

```
roslaunch lio_sam relocalization.launch
rosbag play park.bag
```

#### 
