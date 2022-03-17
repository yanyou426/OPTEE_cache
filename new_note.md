# 3.13 SGX cache

## 摘要

现有的SGX 技术是无法抵抗CPU Cache 侧信道攻击的，本论文对运行在安全区内部的AES加密程序实施了攻击。攻击的目标是OpenSSL 的AES 优化实现，是最后一轮查找表访问驱动的已知密文攻击，利用Prime+Probe 侧信道手段来判断最后一轮是否访问了查找表的某些表项，使用Neve&Seifert 消除法，可以恢复出最后一轮的轮密钥，从而得知AES 的加密密钥。

## 常见cache侧信道攻击方法

侧信道攻击的目标是**判断被害者程序有没有访问某一个确切地址的内存数据**，假设这个确切地址是addr，作为cache 攻击的条件，攻击者可以获取addr 的值。下面是几种判断受害者程序有没有访问addr 的cache 攻击手段。

### evict+reload

第一步，攻击者把addr 地址处的数据填入缓存
		第二步，Time：触发执行受害者进程并计时
		第三步，Evict：将addr 处的数据从缓存中驱逐
		第四步，Time: 重新触发执行受害者进程并计时
		如果第四步花费的时间比第二步长，则说明受害者进程很有可能使用了addr
处的数据。

### flush+reload

第一步，Flush：攻击者使用clflush 指令把addr 处的数据从缓存中驱逐.

第二步，攻击者触发受害程序的执行。
		第三步，Reload：攻击者访问addr 处的数据，并计时。
		攻击者衡量第三步所花费的时间，如果**超过一个阈值**，则说明addr 处的数据未被放置到cache 里，即受害者进程未使用到addr 处的数据。



### flush+flush

攻击的基础是：如果addr处的数据未被放在cache 中，则**使用clflush 指令逐出addr 处的数据时，指令执行的时间会短一些。**
		Flush+Flush 攻击与Flush+Reload 的应用场景几乎相同，攻击步骤也只是第三步稍有不同，Flush+Flush 攻击第三步使用clflush 指令逐出addr 处的数据并计时，如果时间超过一个阈值，就认为攻击者访问了addr 处的数据。



### prime+probe

Prime+Probe 攻击的对象是addr 所映射的缓存组，例如addr 所在的缓存组的序号是s，则攻击步骤如下：
		第一步，Prime：攻击者用自己的数据填满缓存组s中的所有缓存项
		第二步：攻击者触发受害者程序运行
		第三步，Probe：攻击者遍历自己在第一步中准备的数据，并且计时。
		如果受害者程序访问了addr 处的数据，则攻击者在Prime 过程中构造的填满缓存组s的数据就会有一个或者多个被驱逐，攻击者Probe 过程花费的时间就会长一些，可以利用时间长短来判断受害者程序有没有访问addr。



## 选择最后一种攻击方法

前三种都必须基于共享内存，而第四种，SGX环境无法抵抗基于内存地址的Prime+Probe攻击。



## 用PP方法之后的排除法

![image-20220317140259877](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220317140259877.png)





# 3.15 装linux系统

因为疫情进不了实验室，快递也没办法及时寄过来，心里肯定是着急的捏。

今天一个小目标，先把linux系统装好吧

## 几个问题：

https://www.bilibili.com/video/BV1554y1n7zv?p=9



### 重启时黑屏：

按照视频装到p8时，笔记本自带键盘没法输入，用了外接键盘才成功安装。注意，语言尽量先选英文！！！

后来重启时出现问题了，ubuntu界面时点击e，在文档中的quite splash后加nomodeset,按fn+f10保存重启

但以上操作是需要每次都这样的，直接进/etc/default里`"quite splash"`后加个 `nomodeset`，重启成功！

应该是英伟达驱动的问题。



### 怎么装中文输入法？

在region&language设置中，点击manage install language，再点击install/remove languages添加简体中文，再点击+号，添加Chinese(pinyin)即可



### 自带键盘失灵问题怎么修改？

/etc/default/grub 在quiet splash后加上i8042.dumbkbd

然后`sudo update-grub`



### 网速很慢问题怎么解决？

目前换了阿里源，/etc/apt/sources.list中为更改过的阿里源,/etc/apt/sources.list.bac为原来的备份。

结束之后`sudo apt-get update`

不知道需要的命令对应的软件包时 `sudo apt-cache search 命令名`





# 3.17 Prime_Count

To cope with these challenges, we need a novel cache attack approach that does not require memory sharing and introduces less noise in the cross-world scenario. In this paper, we leverage an overlooked ARM Performance Monitor Unit (PMU) feature named “L1/L2 cache refill events” and design a Prime+Count technique that only cares about how many cache sets or lines have been occupied instead of determining which cache sets have been occupied as in Prime+Probe. The coarser-grained approach significantly reduces the noise introduced by the pseudo-random replacement policy and world switching. Even though some performance counters in PMU, such as cycle counter, have been used to carry out and detect cache-based side-channel attacks in the ARM and Intel architecture [10, 42], to the best of our knowledge it is novel to use “L1/L2 cache refill events” to perform attacks.

[10] Marco Chiappetta, Erkay Savas, and Cemal Yilmaz. 2016. Real time detection of
cache-based side-channel attacks using hardware performance counters. Applied Soft Computing 49 (2016), 1162–1174.

[42] Ning Zhang, Kun Sun, Deborah Shands, Wenjing Lou, and Y Thomas Hou. 2016.
TruSpy: Cache Side-Channel Information Leakage from the Secure World on ARM Devices. https://eprint.iacr.org/2016/980.pdf. (2016).



# 3.17 ARMageddon: Cache Attacks on Mobile Devices









