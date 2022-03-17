## 文件结构

/bin        二进制文件，系统常规命令

/boot       系统启动分区，系统启动时读取的文件

/dev        设备文件

/etc        大多数配置文件

/home       普通用户的家目录

/lib        32位函数库

/lib64      64位库

/media      手动临时挂载点

挂载，指的就是将设备文件中的顶级目录连接到 Linux 根目录下的某一目录（最好是空目录），访问此目录就等同于访问设备文件

/mnt        手动临时挂载点

如何以命令行方式访问u盘数据？

```
cd /dev 查看u盘名称
cd /mnt 
(sudo) mkdir usb
sudo mount -t vfat /dev/sdb1(设备名) /mnt/usb
cd /mnt/usb 查看u盘文件
使用完毕后卸载设备： sudo umount /dev/sdb1
```



/opt        第三方软件安装位置

/proc       进程信息及硬件信息

/root       临时设备的默认挂载点

/sbin       系统管理命令

/srv        数据

/var        数据

/sys        内核相关信息

/tmp        临时文件

/usr        用户相关设定



## 基础操作

立刻关机

```
shutdown -h now 
power off
```

立刻重启

```
shutdown -r now 
reboot
```

帮助命令

```
ifconfig --help
```

命令说明书

```
man shutdown (q退出)
```

列表查看所有文件（更全面）

```
ls -l
```

删除

```
rm filename
rm -rf directory
rm -rf * 清空当前目录下的所有文件
```

重命名、剪切

```
mv oldname newname
mv -r /tmp/tool /opt 剪切目录
```

拷贝目录

```
cp -r old new
```

搜索目录

```
find /bin -name 'a*' 查找bin目录下的所有以a开头的文件或者目录
```



## 文件操作

新增

```
touch a.txt
```

查看

```
cat a.txt 查看文件最后一屏幕内容
less a.txt 上下键查看页面，q退出查看
tail -100 a.txt 显示最后100行 ctrlC退出查看
```

更改权限

```
chmod +x a.txt
chnmod 777 a.txt (1+2+4=7) 说明赋予所有权限
```



## 打包与解压

```
tar -zcvf a.tar file1 file2... 打包并压缩
tar -zxvf a.tar -C /path
unzip test.zip
unzip -l test.zip 查看*.zip文件的内容
```



## 其他

```
find . -name "*.c" //将目前目录及其子目录下所有延伸档名是 c 的文件列出来
find . -type f //将目前目录其其下子目录中所有一般文件列出

locate /etc/sh 搜索 etc 目录下所有以 sh 开头的文件
locate -i ~/r 忽略大小写搜索当前用户目录下所有以 r 开头的文件

grep -A 3 -r string * 在当前文件夹下递归查询包含指定字符串string的文件，且输出那行之后的三行
grep –e "正则表达式" 文件名
top 显示当前系统中占用资源最多的一些进程
uname -a 显示一些重要的系统信息，例如内核名称、主机名、内核版本号、处理器类型之类的信息
ps -ef //查看所有正在运行的进程

```









