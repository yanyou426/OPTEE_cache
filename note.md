

# 学习进度





# 11.14-11.20 书

## 密码设备攻击

被动攻击 主动攻击

入侵式攻击 半入侵式攻击 非入侵式攻击

侧信道攻击：被动型非入侵式攻击，最重要的有三类，计时攻击、能量分析攻击、电磁攻击

## 能量分析攻击

基本思想：通过分析密码设备的能量消耗获得其密钥

基于：密码设备的瞬时能量消耗依赖于所处理的数据和所执行的操作

## 能量消耗

CMOS元件能量消耗：

$P = P_{stat} + P_{dyn}$

$P_{stat} = I_{leak} * V_{DD}$

$P_{dyn}$:元件中负载电容需要充电+元件输出信号转换时产生的瞬时短路电流

## 适用于攻击者的能量模型

汉明距离模型通常刻画总线和寄存器的能量消耗，攻击者可以在不拥有设备网表的情况下

其余情况通常使用汉明重量模型

能量迹中的噪声=电子噪声（相同参数对设备多次测量时能量迹中不同的波动）+转换噪声（影响因素：元件与示波器间的带宽，设备的时钟频率）

## 能量迹的组成

$P_{total} = P_{op} + P_{data} + P_{el.noise} + P_{const}$

其中，电子噪声$P_{el.noise}$服从正态分布

$P_{total}$与正在处理的数据的汉明重量成反比

对于大多数设备而言，若被处理的数据服从均匀分布，则$P_{data}$,$P_{op}$均可以用正态分布近似表示

## 能量迹单点泄露

$P_{op} + P_{data} = P_{exp} + P_{sw.noise}$

$P_{exp}$为攻击场景中可以利用的能量消耗分量

$P_{sw.noise}$为无法利用的转换噪声

利用$P_{exp}$,$P_{sw.noise}$,$P_{el.noise}$定义信噪比SNR

**能量迹中的尖峰是与能量分析攻击最相关的点，因为此时SNR达到最大值，暴露信息最多**

# 11.21-11.26 书

## 简单能量分析SPA

前提：攻击者能够监测设备的瞬时能量消耗，设备中的密钥必须对能量消耗有间接或直接的影响

### 直观分析

以AES加密为例

通过分析能量迹中的尖峰或者模式，可以确定出不同类型的数据传送指令

### 模板攻击

#### 前提

能量消耗依赖于设备正在处理的数据

#### 阶段

1.使用**多元正态分布**对能量消耗特征进行刻画

2.利用该特征实施攻击

#### 具体过程

1.对每组数据和密钥$(d_i,k_j)$计算均值与协方差矩阵$(m,C)$，即模板h
$$
h_{d_i,k_j}=(m,C)_{d_i,k_j}
$$
2.对已知的能量迹t，计算和每一组$(d_i,k_j)$的概率密度

3.概率值的大小反映了模板h与t的匹配程度

#### 模板构建策略

1.数据和密钥对模板构建

用于构建模板的特征点就是能量迹中与密钥对相关的所有点

2.中间值模板构建

为$f(d_i,k_j)$构建模板

3.基于能量模型的模板构建

为同一汉明重量的数值构建相同的模板，如构建S盒输出的模板时，创建9个即可

#### 模板匹配

![image-20211124105454585](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20211124105454585.png)

#### MOV指令模板攻击示例

能量迹上点的高度与被处理字节的汉明重量成反比

### 碰撞攻击

对于两个不同输入，会有部分密钥值导致中间的某个输出值相等，即发生中间值碰撞，以此减小密钥搜索空间

## 差分能量分析DPA

知道设备中执行的是何种密码算法即可，分析固定时刻时能量消耗与被处理数据之间的依赖关系，**将能量迹看作被处理数据的某个函数**

### 基本步骤

1.选择中间值$f(d,k)$，其中d为明文或密文，k为密钥的一小部分

2.测量能量消耗：对D个数据测量长为T的能量迹，最后即获得一个D*T的矩阵能量迹**T**

3.计算假设中间值

<img src="C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20211125174054428.png" alt="image-20211125174054428" style="zoom:50%;" />

4.使用能量模型将中间值映射为能量消耗：**V**->**H**

<img src="C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20211125174203910.png" alt="image-20211125174203910" style="zoom:50%;" />

5.将假设能量消耗矩阵和实际能量迹矩阵比较，每一列都分别计算相似度，形成K*T的矩阵R，且值越大，匹配度越高

6.最大值所在行ck为密钥索引，所在列ct为为时刻

## CPA相关能量分析

选取n个明文，用示波器采集加密过程中的能量迹trace，即电压v

猜测密钥k->异或后的x->S盒输出的y->哈希值h

计算v和h的相关系数

![image-20211125190816633](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20211125190816633.png)



# 11.30-12.4 DPA曲线集

阅读了一下论文和前期工作，打开了trc文件

# 12.10-12.16 DPAcontest

整理cpa基础理论知识，和师兄开会理了一下大致流程

# 1.7-1.9 CPA攻击密钥

首先确定采样点435002，使用10000个曲线进行攻击

将index文件和曲线集处理后存储为npy二进制文件，方便后续处理

同时读取mask文件，里面存储了16个mask值，是aes加密后续需要用到的



处理能量迹曲线

```python
struct.unpack(fmt, string)
```

解包，返回一个由数据string得到的一个元组tuple，其中len(string) 必须等于 calcsize(fmt)，这里面涉及到了一个calcsize[函数](http://www.php.cn/wiki/145.html)。struct.calcsize(fmt)：这个就是用来计算fmt格式所描述的结构的大小。I即为unsigned int

```python
data_points_num = struct.unpack("<I", content[base_offset + 116: base_offset + 116 + 4])[
    0]
```

```python
point = struct.unpack("b", bytes([content[data_offset + first_points_ind + i]]))[0]
#每8位作为一个整数解包存储，-128-127
```

将前off个曲线作为数组存储至00000.npy文件中，后面直接读取



```python
real_key = bytes.fromhex(showed_key[0])
```

b'l\xec\xc6\x7f(}\x08=\xeb\x87f\xf0s\x8b6\xcf\x16N\xd9\xb2F\x95\x10\x90\x86\x9d\x08(].\x19;'

```python
print("密钥为：")
print([int(b) for b in real_key])
```

密钥为：
[108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207, 22, 78, 217, 178, 70, 149, 16, 144, 134, 157, 8, 40, 93, 46, 25, 59]

```python
round_key = aes_cipher.createRoundKey(expandedKey, 0) # 第一轮轮密钥，即前16个数字
# print(round_key)
first_key_byte = round_key[0] # 0x6c
print(first_key_byte) #108
first_round_key = [round_key[i * 4 + j] for j in range(4) for i in range(4)]
print('The First Key byte: ', first_key_byte)
print('The First Round Key: ', first_round_key)
```

第一轮轮密钥是咋生成的呢？

i = 0, j = 0/1/2/3, [round[0,1,2,3]]

生成一个1*16的第一轮轮密钥

The First Round Key:  [108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207]



下面计算理论泄露值即HW的那个矩阵，统一先取第一个字节，最后也就是先攻击出第一个字节的密钥

```python
masked_state_byte_leak = np.array([HW[SBOX[p[0] ^ first_key_byte] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])
```

想计算后15字节的密钥，只需要改p后面的**索引值**

密钥（仅攻击第一轮密钥）

6cecc67f287d083d eb8766f0738b36cf

164ed9b246951090869d08285d2e193b



**最后结果，提取前500条trace进行攻击。**

**********处理index_file基础信息*************
Reading DPA index file from 0 to 499
Saving DPA index file done.
plaintxt/ciphertxt samples:  [['448ff4f8eae2cea393553e15fd00eca1' 'f71e9995e754e9f711b4027106a72788']
 ['d0edb7612c4dc8aa42358571649af40c' 'f0fbbbb6e7d2befb7b947e9250fcd754']]
offset samples:  ['8' '8' 'b' '5' '4']
AES key: 6cecc67f287d083deb8766f0738b36cf164ed9b246951090869d08285d2e193b



**********处理trace曲线集,以trace1为例*************
The very first 10 power sample: [-6 -5 -4  0  0 -5 -1 10 13  9]



**********处理前1000个曲线并存储至00000.npy文件*************
**********已提前运行过,此处不展示具体过程**********************



密钥为：
[108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207, 22, 78, 217, 178, 70, 149, 16, 144, 134, 157, 8, 40, 93, 46, 25, 59]
The First Key byte:  108
The First Round Key:  [108, 236, 198, 127, 40, 125, 8, 61, 235, 135, 102, 240, 115, 139, 54, 207]



Reading traces from 0 to 500 done.



*************计算16个字节的泄露值和trace的相关性**************
The most possible key is:  6c
The most possible key is:  6cec
The most possible key is:  6cecc6
The most possible key is:  6cecc67f
The most possible key is:  6cecc67f28
The most possible key is:  6cecc67f287d
The most possible key is:  6cecc67f287d8
The most possible key is:  6cecc67f287d83d
The most possible key is:  6cecc67f287d83deb
The most possible key is:  6cecc67f287d83deb87
The most possible key is:  6cecc67f287d83deb8766
The most possible key is:  6cecc67f287d83deb8766f0
The most possible key is:  6cecc67f287d83deb8766f073
The most possible key is:  6cecc67f287d83deb8766f0738b
The most possible key is:  6cecc67f287d83deb8766f0738b36
The most possible key is:  6cecc67f287d83deb8766f0738b36cf



![cparesult](C:\Users\颜又和瓜蛋\Desktop\CPA\cparesult.png)

运行时间大概一个半小时，好久。。。

# 1.10-1.11 CPA代码知识点

## 计算pearson相关性

```python
def pearson(X: np.ndarray, Y: np.ndarray):
    if X.shape[0] != Y.shape[0]:
        print("X and Y have wrong dimension")
        return
    # X: N*1, Y:N*M
    # 也就是说 X是一个字节加密后的理论值 Y是选的所有trace
    mean_X = X.mean(axis=0)
    #axis是几，代表哪一维度被压缩成1
    mean_Y = Y.mean(axis=0)  # mean of Y by column
    #此时，x被压缩成一个值，y被压缩成一条横线
    XX = (X - mean_X).reshape(X.shape[0])
    #XX是减完平均值后的n*1
    YY = Y - mean_Y
    #YY是减完平均之后的n*m
    r_X = np.sqrt(np.sum(np.square(XX), axis=0))
    #r_X是XX所有值平方的和再开更，最后就是一个数
    r_Y = np.sqrt(np.sum(np.square(YY), axis=0))
    #r_Y是YY所有值平方后，压缩成一行的求和，再开更，是m*1,但是横着的
    sum_XY = np.matmul(YY.T, XX)
    #m*n，n*1  得m*1的横着的矩阵
    r = sum_XY / (r_X * r_Y)
    # 前面这个被除数是 m*1
    # 后面这个除数是对应项相乘，最后得到m*1
    return r # 返回 m*1
```

np.matmul(a, b)

若都是两维的，做普通的矩阵相乘 

## code整体思路

读取index_file,将明密文、offset、密钥分别保存到npy文件中

另外读取一下mask的数组 16个数字

处理trace文件，去掉前面的357个字节，最后整合成一个trace_num*435002的一个np数组，作为raw_trace

读取明文和对应的偏移量，我这边只选了上面保存好的npy文件里的前500个trace，要不然时间有点久，下面的二层循环一共花了一个半小时左右。

```python
 for plain_ind in range(16): # 对每个字节的明文的密钥都需要攻击
    max_corr_k = np.zeros(256)  # 记录256个k值时，最大的相关性，最后那个最大的相关性的k有最大可能成为这一字节的密钥
    for k in range(256):
        #计算每个明文的第plain_ind字节的泄露模型也就是hw值，是一个列向量 500 * 1
        masked_state_byte_leak1 = np.array(
            [HW[SBOX[p[plain_ind] ^ k] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])

        # 相关性寻找，计算500条trace的每一列，与理论泄露值进行相关系数的计算，取出相关系数最大的那些点的索引
        s1_corr_rank = np.zeros(sample_length)
        candidate_traces = raw_traces[:analysis_trace_num]
        s1_corr_rank += correlation(masked_state_byte_leak1[:analysis_trace_num], candidate_traces)
        #此时s1_corr_rank就是一个435002 * 1的列向量，代表了相关系数的大小，相关系数最大的那一列在trace中的体现就是那个时刻发生了s盒的输出
        s1_ind = s1_corr_rank.argsort()[-most_corr_point:][::-1] # 选中前5大的相关值的ind
        max_corr_k[k] += s1_corr_rank[s1_ind[0]] # s1_ind[0]就是最大相关性的索引值，max_corr_k[k]就是当key为k时的相关性
        # plt.plot(s1_corr_rank)
        # plt.show()

    max_corr_ind = max_corr_k.argsort()[-most_corr_point:][::-1]  # 将最大的几个相关性的key的索引值也就是key本身排列出来
    result += hex(max_corr_ind[0])[2:].zfill(2) #有个特殊值08 需要填满0 否则少一位 max_corr_ind[0]就是最有可能的key
    print("The most possible key is: ", result)
```







# 1.12-1.14 Template Attack 尝试

## 一字节key思路

将trc文件转化成npz文件，转化后取以下两个文件

```python
trace_num = 10000 # 共10000条plaintext 有1000个采样点
raw_traces = target["trace"]
plaintexts = target["crypto_data"]
```

对raw_traces进行标准化

```python
def standardize(traces):
    mu = np.average(traces, 1) # 在列上压缩 平均值
    std = np.std(traces, 1) # 求标准差
    if np.min(std) != 0:
        traces = (traces - mu.reshape(mu.shape[0], 1)) / std.reshape(mu.shape[0], 1)
        # 将平均值变成一列（其实本来就是一列），被原值减，再除标准差
    return traces
```

先进行了**PCA主成分分析**操作，将trace进行了采样点的压缩

```
pca = PCA(traces, explain_ratio=0.95)
traces = pca.proj(traces)
```

具体操作是：

```python
class PCA:
    '''
    A general PCA class
    '''
    proj_matrix = None

    def __init__(self, X, explain_ratio=0.95):
        cov_matrix = np.cov(X.T)#10个特征 每个特征5个观测值
        # print(cov_matrix.shape) # 10 * 10
        eigen_values, eigen_vectors = np.linalg.eig(cov_matrix) # 特征值 特征向量
        # print(eigen_values.shape)
        # print(eigen_vectors.shape)
        esum = np.sum(eigen_values) # 矩阵的迹
        # print(esum)
        variance_explained = np.zeros(eigen_values.shape[0]) # 10 * 1
        for i, v in enumerate(eigen_values):
            variance_explained[i] = v / sum(eigen_values) #每个值就是具体的特征值除总和
        # print(variance_explained)
        cumulative_variance_explained = np.cumsum(variance_explained) #不规定轴方向，即当成一维数组一直累加
        # print(cumulative_variance_explained)
        # print(np.where(explain_ratio <= cumulative_variance_explained)) # 挑选出叠加矩阵里大于ratio的下标
        self.proj_matrix = eigen_vectors[:, :max(1, np.where(explain_ratio <= cumulative_variance_explained)[0][0])]
        # print(self.proj_matrix)
        print("Dimension reduced from %d to %d" % (len(eigen_values), self.proj_matrix.shape[1]))
        return

    def proj(self, x):
        # print(np.dot(x, self.proj_matrix))
        # print('实部：', np.real(mat))
        # print('虚部：', np.imag(mat))
        return np.real(np.dot(x, self.proj_matrix))
```

然后将主成分分析后的10000*7分成train组和attack组

```python
ta = TA(traces=train_tr, plain_texts=train_pt, real_key=train_key, num_pois=5)
mean_matrix, cov_matrix, guessed = ta.attack(attack_tr, attack_pt)
```

第一行是 对每个hw构建均值向量和协方差矩阵

第二行是根据构建的模型，测试剩下的200个trace，每个key的可能性，可能性最大的就是最有可能的key



## 攻击offset

我找了一个模板攻击的代码，我看他就是10000条里面，提取9800条的特征构建平均值和协方差矩阵，那在构建的时候需要将offset看成已知的吗？

然后200条作为攻击曲线猜测key，

```python
for j, trace in enumerate(traces): # 把每个trace挑出来
    # Grab key points and put them in a small matrix
    a = [trace[poi] for poi in self.pois]
    # print(a)
    # Test each key
    for k in range(256):
        # Find leak model coming out of sbox
        mid = self.leak_model[SBOX[plaintext[j] ^ k]]
        ......
```

**把offset构建16个模板，先攻击offset，然后再用攻击好的offset去攻击key**



# 1.15-1.17 模板攻击：假设已知offset，先攻击出key ✔

Q1：trace有很多采样点，在提取特征点时，采用主成分分析，其中计算矩阵的迹时，矩阵会很大

![image-20220118114312938](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220118114312938.png)

Q2：在攻击第一字节的key时，即使假设知道是在采样点为228403处相关性最大，我先将225000-240000的trace进行主成分分析，这时矩阵大小15000*15000是可以计算的，这样子key能成功攻击出，但一是速度非常慢，二是在已知攻击时间点的情况下攻击本身就是不合理的。

Q3：上面两个问题都是特征提取时用的PCA这个方法不合理造成的，师兄有没有推荐的其他特征提取的方法？（我在特征提取这方面只接触过PCA ，太菜了TAT）

**用相关性计算**？先挑出相关性高的列数，在他周围进行PCA，再进行模板攻击



# 1.18-1.1.23 配置服务器环境  完成模板（dpa？）攻击

## 模板攻击（？又疑似dpa攻击）（dpav4-dpa4.1-dpa4.1.py）



```python
first_byte_leak = np.array([HW[p[1] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])#记录第二字节的偏移
masked_state_byte_leak = np.array([HW[SBOX[p[0] ^ first_key_byte] ^ mask[(o + 1) % 16]] for p, o in zip(plaintext, mask_offset)])#第一字节
```

再分别画出相关性，并取出相关性最大的前20个点

- 根据等式𝑥1⊕𝑚1⊕𝑆𝐵𝑂𝑋[𝑥0⊕𝑘0]⊕𝑚0=x1⊕SBOX[x0⊕k0]可知
  - 需要构建𝑥1⊕𝑚1与𝑆𝐵𝑂𝑋[𝑥0⊕𝑘0]⊕𝑚1泄漏点之间的联系
  - 使用曲线点之间的能量差值进行联系

```
power_diff = [] # attack_trace_num * 20 list
for trace in raw_traces[-attack_trace_num:]: #在raw选取最后的需要攻击的trace中
    trace_leak_point = [abs(trace[id1] - trace[id2]) for id1, id2 in zip(s1_ind, x1_ind)] #在上面计算的最有关联的两对点中，记录差值，有20个数
    power_diff.append(trace_leak_point)
#最后有attack_num组最大差值的数，存储了与目标泄露点相关性最强的点之间的能量差值
```

遍历每个key，对每个key，统计x1⊕SBOX[x0⊕k0]的泄露值candidate_leak（这是理论的）

​						并对20个可能的候选点记录power_diff[:,leak_point]和candidate_leak的相关性

​						最后记录此时这个key在所有候选点时每个的可能性

```
possible_key, possible_loc = np.where(key_rank == np.max(key_rank))
#np.max不带参数应该就是记录每行的最大值，也就是每个key的最大可能性
#而等于key_rank返回的就是（key，那个纵坐标loc）
```

最后plt.plot(key_rank[:, possible_loc[0]])画出的是横坐标为255个key，纵坐标为每个key对应的最大可能性

## 配置jupyterlab环境

![preview](https://pic1.zhimg.com/v2-25f2c58b8fca04d70a8c635824163cf4_r.jpg)

首先是个快速配置环境的小tip

结果

![image-20220119162058155](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220119162058155.png)

还好最后搞好了555

**注意**：每次进去都需要activate那个环境

## 使用conda进行python的环境管理

#### 创建python环境

```
conda create -n <环境的名字> python=<python的版本>
```

例如，`conda create -n science python=3.8`

#### 使用python环境

```
conda activate <环境的名字>
```

#### 退出python环境

```
conda deactivate <环境的名字>
```

#### 安装库

```
conda install <库的名字>[=<版本号>]
```

例如，`conda install numpy`或者`conda install cudatoolkit=10.2`

#### 集成到jupyterhub

jupyterhub在`https://yuhangji.cn/jupyter/`，通过系统用户名和密码登录，登录之后可以在自己home目录下新建jupyter notebook，并选择python解释器。

要使用本地的python环境，首先需要为本地python环境创建jupyter kernel，

```
conda activate <环境的名字>
conda install ipykernel
python -m ipykernel install --user --name <环境的名字，可以与上面的不一样>
```

然后就可以在jupyterhub里面找到本地安装的kernel了。

#### 其他

* 系统内存有160GB，有两个2080ti可以使用

* 虽然没有sudo权限，但是每个用户都默认拥有创建和管理docker的权限，并且docker可以设置GPU直连，有需要折腾GPU环境的可以进docker折腾。

* 在`/data/loccsftp/upload/`有三个DPAv4.1的zip包，不需要再下载



![image-20220119162849611](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220119162849611.png)



## TA攻击理论知识

模板描述曲线上关键点的多元分布，比如某一密钥猜测对应的概率密度较小，那它大概率是错误的。

![image-20220120155021208](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220120155021208.png)

在一条曲线中选择3-5个重要的点作为POIs，将维度缩减为3-5维

那么如何选取点？差值求和。对所有曲线在t=i时的点求平均，两两作差，并对差值求和，且尖峰周围的n个点要舍弃，直到选够点

![image-20220120160012602](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220120160012602.png)

对每个候选项，建立均值向量和协方差矩阵

应用模板：将剩余曲线的POI代入概率密度函数。得到候选项*曲线数的概率矩阵，横着相乘即为每个key的概率。![image-20220120160801052](C:\Users\颜又和瓜蛋\AppData\Roaming\Typora\typora-user-images\image-20220120160801052.png)



# 2.4 OPTEE设计架构

## 介绍

optee = client (普通世界用户空间的客户端API) + linux kernel device driver (控制普通世界用户空间和安全世界通信的设备驱动) + os (运行在安全世界的可信操作系统)



## 原理

安全世界和普通世界通过SMC异常来通信

![preview](https://pic1.zhimg.com/v2-87f7944e1ad728438f4c459a69f62be8_r.jpg)

CA调用函数部分，发起一次调用请求

TEE_client作用

TEE-supplicant作用

OP-TEE drive

monitor mode如何接收中断请求

OP-TEE OS如何处理请求

TA实现功能并返回结果



## 完整调用流程

在userspace层面调用CA接口后会触发system call操作，系统调用会将Linux陷入内核态，此时处于kernel space，然后根据传入的参数找到对应的tee driver

tee driver到secure moniter状态（SMC），随即进入tee kernel

tee os接管剩下的操作，首先获取从CA传过来的数据，解析出TA的UUID，并查找对应的TA image是否被挂载到TEE OS中，若没有，则与常驻在Linux中的tee_supplicant进程通信，利用它从文件系统中获取TA image文件，并传递给TEE OS，加载该image。然后，TEE OS会切换到TEE userspace态，并将CA传递过来的其他参数传给具体的TA process。TA process解析出commond ID，并根据这个来做具体的操作。



# 2.23 树莓派

## 硬件

![RaspberryPi_3B](C:\Users\hky\Downloads\RaspberryPi_3B.svg)

Rasberry Pi 3 Model B

以Linux内核的操作系统

| 规格         | 树莓派 3                                                     |
| ------------ | :----------------------------------------------------------- |
| **处理器**   | 四核 1.2GHz 博通 BCM2837                                     |
| **RAM**      | 1 GB                                                         |
| **蓝牙**     | BLE                                                          |
| **USB端口**  | 4 x USB 2.0                                                  |
| **无线连接** | 是                                                           |
| **显示端口** | 1 x HDMI，1 x DSI                                            |
| **电源**     | microUSB 和 GPIO，高达 2.5 A                                 |
| **SoC**      | [Broadcom](https://zh.wikipedia.org/wiki/Broadcom) BCM2837（[CPU](https://zh.wikipedia.org/wiki/CPU)，GPU DSP和[SDRAM](https://zh.wikipedia.org/wiki/SDRAM)、[USB](https://zh.wikipedia.org/wiki/USB)） |
| **GPU**      | [Broadcom](https://zh.wikipedia.org/wiki/Broadcom) VideoCore IV[[27\]](https://zh.wikipedia.org/wiki/树莓派#cite_note-hq-qa-27), [OpenGL ES](https://zh.wikipedia.org/wiki/OpenGL_ES) 2.0, [1080p](https://zh.wikipedia.org/wiki/1080p) 30 [h.264](https://zh.wikipedia.org/wiki/H.264)/[MPEG-4](https://zh.wikipedia.org/wiki/MPEG-4) [AVC](https://zh.wikipedia.org/wiki/AVC)高清解码器 |



## 操作系统

https://zhuanlan.zhihu.com/p/147061445  树莓派系统+optee





# 2.25 op-tee AES

https://icyshuai.blog.csdn.net/article/details/71517567

![image-20220228151505376](C:\Users\hky\AppData\Roaming\Typora\typora-user-images\image-20220228151505376.png)



ta/sub.mk  将TA中所有的.c添加到编译文件中

ta/include/aes_ta.h:定义该TA程序的UUID，后续被host/main.c调用

ta/user_ta_header_defines.h:将上面这个文件include到该文件中以便获取UUID

ta/makefile:将BINARY改为UUID的值($()用来做命令替换，比如显示年月日)

**ta/aes_ta.c**:具体实现代码：：

```c
TEE_Result TA_CreateEntryPoint(void)
{
	/* Nothing to do */
	return TEE_SUCCESS;
}


//回应TEEC_OpenSession，在安全世界中创建一块区域
TEE_Result TA_OpenSessionEntryPoint(uint32_t __unused param_types,
					TEE_Param __unused params[4],
					void __unused **session)
{
	struct aes_cipher *sess;//新建一个结构体

	/*
	 * Allocate and init ciphering materials for the session.
	 * The address of the structure is used as session ID for
	 * the client.
	 */
	sess = TEE_Malloc(sizeof(*sess), 0);
	if (!sess)
		return TEE_ERROR_OUT_OF_MEMORY;

	sess->key_handle = TEE_HANDLE_NULL;
	sess->op_handle = TEE_HANDLE_NULL;

	*session = (void *)sess;
	DMSG("Session %p: newly allocated", *session);

	return TEE_SUCCESS;
}

//被TEEC_InvokeCommand调用
//cmd决定调用具体哪个功能
TEE_Result TA_InvokeCommandEntryPoint(void *session,
					uint32_t cmd,
					uint32_t param_types,
					TEE_Param params[4])
{
	switch (cmd) {
	case TA_AES_CMD_PREPARE:
        //处理TA_AES_CMD_PREPARE，检查params[0-2].value.a是否属于对应的属性
        //TEE_AllocateOperation() 分配一个进行密码操作的操作句柄，并设定算法类型和模式
        //TEE_AllocateTransientObject() 分配一个未初始化的临时object空间
        //TEE_InitRefAttribute() 使用key初始化attr
        //TEE_PopulateTransientObject(TEE_ObjectHandle sess->key_handle,const TEE_Attribute &attr) 将attr赋值到key_handle中
        //TEE_SetOperationKey(TEE_OperationHandle op_handle,TEE_ObjectHandle sess->key_handle) 将存放密钥的object中的相关内容保存到操作句柄中
		return alloc_resources(session, param_types, params);
	case TA_AES_CMD_SET_KEY:
        //TEE_SetOperationKey(sess->op_handle, sess->key_handle)
		return set_aes_key(session, param_types, params);
	case TA_AES_CMD_SET_IV:
        //TEE_CipherInit(TEE_OperationHandle sess->op_handle, const void iv,uint32_t iv_sz) 使用初始化向量初始化对称加密操作
		return reset_aes_iv(session, param_types, params);
	case TA_AES_CMD_CIPHER:
        // TEE_CipherDoFinal(TEE_OperationHandle sess->op_handle,const void params[0].memref.buffer, uint32_t params[0].memref.size,void params[1].memref.buffer, uint32_t &params[1].memref.size)
        //完成加解密操作
		return cipher_buffer(session, param_types, params);
	default:
		EMSG("Command ID 0x%x is not supported", cmd);
		return TEE_ERROR_NOT_SUPPORTED;
	}
}


//关闭会话
void TA_CloseSessionEntryPoint(void *session)
{
	struct aes_cipher *sess;

	/* Get ciphering context from session ID */
	DMSG("Session %p: release session", session);
	sess = (struct aes_cipher *)session;

	/* Release the session resources */
	if (sess->key_handle != TEE_HANDLE_NULL)
		TEE_FreeTransientObject(sess->key_handle);
	if (sess->op_handle != TEE_HANDLE_NULL)
		TEE_FreeOperation(sess->op_handle);
	TEE_Free(sess);
}

void TA_DestroyEntryPoint(void)
{
	/* Nothing to do */
}
```



host/makefile:主要修改BINARY变量和OBJ变量，如果CA部分的代码不止一个.c文件，则需要将所有的.c文件编译生成的.o文件名称(main.o)添加到OBJS变量中，而BINARY变量就是编译完成之后生成的Binary的名称
**host/main.c：**：完成主要流程的代码

```c
int main(void)
{
	struct test_ctx ctx;
	//设置密钥 初始向量 
	char key[AES_TEST_KEY_SIZE];//key有16字节
	char iv[AES_BLOCK_SIZE];//初始化向量
	char clear[AES_TEST_BUFFER_SIZE];//明文
	char ciph[AES_TEST_BUFFER_SIZE];//密文
	char temp[AES_TEST_BUFFER_SIZE];//从密文解密后得到的明文

    //prepare函数中，
    //TEEC_InitializeContext(NULL, &ctx->ctx)
    //TEEC_OpenSession(&ctx->ctx, &ctx->sess, &uuid,TEEC_LOGIN_PUBLIC, NULL, NULL, &origin);此时安全世界会加载TA，根据uuid返回一个session结构体，该结构体记录了TA被加载到optee的具体位置和入口点函数的地址，使用invokecommand时，optee根据session结构体的参数找到TA的入口点
	printf("Prepare session with the TA\n");
	prepare_tee_session(&ctx);

    //prepare_aes函数中
    //op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INPUT,TEEC_VALUE_INPUT,TEEC_VALUE_INPUT,TEEC_NONE);传入aes操作的相关参数，加密方式CTR,128位密钥，以及看是加密还是解密
    //TEEC_InvokeCommand(&ctx->sess, TA_AES_CMD_PREPARE,&op, &origin);
    //TA_AES_CMD_PREPARE是实际要TA执行的操作 commond:0.同时绑定后面的op，可以进行这个实际操作中的数值设置
	printf("Prepare encode operation\n");
	prepare_aes(&ctx, ENCODE);

    //TEEC_InvokeCommand(&ctx->sess, TA_AES_CMD_SET_KEY,&op, &origin);
    //TA_AES_CMD_SET_KEY commond:1
	printf("Load key in TA\n");
	memset(key, 0xa5, sizeof(key)); /* Load some dummy value */
	set_key(&ctx, key, AES_TEST_KEY_SIZE);
    
	//TA_AES_CMD_SET_IV COMMOND:2
	printf("Reset ciphering operation in TA (provides the initial vector)\n");
	memset(iv, 0, sizeof(iv)); /* Load some dummy value */
	set_iv(&ctx, iv, AES_BLOCK_SIZE);
	
    //TA_AES_CMD_CIPHER COMMOND:3
	printf("Encode buffer from TA\n");
	memset(clear, 0x5a, sizeof(clear)); /* Load some dummy value */
	cipher_buffer(&ctx, clear, ciph, AES_TEST_BUFFER_SIZE);
 
	//解密
	printf("Prepare decode operation\n");
	prepare_aes(&ctx, DECODE);

	printf("Load key in TA\n");
	memset(key, 0xa5, sizeof(key)); /* Load some dummy value */
	set_key(&ctx, key, AES_TEST_KEY_SIZE);

	printf("Reset ciphering operation in TA (provides the initial vector)\n");
	memset(iv, 0, sizeof(iv)); /* Load some dummy value */
	set_iv(&ctx, iv, AES_BLOCK_SIZE);

	printf("Decode buffer from TA\n");
	cipher_buffer(&ctx, ciph, temp, AES_TEST_BUFFER_SIZE);

	/* Check decoded is the clear content */
	if (memcmp(clear, temp, AES_TEST_BUFFER_SIZE))
		printf("Clear text and decoded text differ => ERROR\n");
	else
		printf("Clear text and decoded text match\n");
	
    //TEEC_CloseSession(&ctx->sess);
	//TEEC_FinalizeContext(&ctx->ctx);
	terminate_tee_session(&ctx);
	return 0;
}
```



总目录下执行build_ta_hello_world.sh脚本文件编译TA CA代码，ta目录下生成ta image文件

修改build/common.mk文件：？

修改build/qemu.mk：添加该TA的编译目标



在build目录下直接执行make -f qemu.mk all



在build目录下运行make run-only或者make -f qemu.mk run-only





# 2.28  具体注释op-tee demo



![img](https://upload-images.jianshu.io/upload_images/11314878-1e80faa54cac2555.jpg?imageMogr2/auto-orient/strip|imageView2/2/w/886/format/webp)

TEEC_InitializeContext：连接OP-TEE drive，建立context，提供CA连接TA的方式

TEEC_OpenSession：load TA，根据UUID建立CA呼叫TA的通道

TEEC_InvokeCommand：控制TA函数，传送指令给TA

TEEC_CloseSession：关闭通道

TEEC_FinalizeContext：移除建立好的背景



（sudo minicom -s

第三个 a modem）



CA部分代码示例：

```cpp
int main(int argc, char *argv[])
{
    TEEC_Result res;
    TEEC_Context ctx;
    TEEC_Session sess;
    TEEC_Operation op;
    TEEC_UUID uuid = TA_NEW_TAPS_UUID;
    uint32_t err_origin;

/* 
 * CA第一个调用，初始化上下文连接至TEE 
 调用TEEC_InitializeContext函数打开op-tee驱动文件，获取到操作句柄并存放到TEE_Context类型的变量中。
 即初始化一个TEEC_Context变量，该变量用于CA和TA之间建立联系
 */
    res = TEEC_InitializeContext(NULL, &ctx);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InitializeContext failed with code 0x%x", res);

/*
 * 开启会话，当与TA端会话建立完成，Log输出
 * 当TA开启新的会话时调用This is Hiro!
   打开一个CA与对应TA之间的一个session，该session用于该CA与对应TA之间的联系，该CA需要连接的TA是由UUID指定的。session具有不同的打开和连接方式，根据不同的打开和连接方式CA可以在执行打开session的时候传递数据给TA，以便TA对打开操作做出权限检查。各种打开方式说明如下：
   TEEC_LOGIN_PUBLIC：不需要提供，也即是connectionData的值必须为NULL
   倒数第二个变量operation:指向TEEC_Operation结构体的变量，变量中包含了一系列用于与TA进行交互使用的buffer或者其他变量。如果在打开session时CA和TA不需要交互数据，则可以将该变量指向NULL
 */
    res = TEEC_OpenSession(&ctx, &sess, &uuid,
                   TEEC_LOGIN_PUBLIC, NULL, NULL, &err_origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_Opensession failed with code 0x%x origin 0x%x",
            res, err_origin);

/*
 * 通过调用它来执行TA中的一个功能，在这种情况下我们增加一个数字。 
 * 命令ID部分的值以及如何解释参数是TA提供的接口的一部分。
 */

/* 组合TEEC_Operation结构体 */
    memset(&op, 0, sizeof(op));

/*
 * 准备参数。 在第一个参数中传递一个值，剩下的三个参数是未使用的。
   初始化TEEC_Operation类型的变量，并根据实际需要借助TEEC_PARAM_TYPES宏来设定TEEC_Operation类型变量中paramTypes成员的值，该值规定传递到OP-TEE中的最多4个变量缓存或者是数据的作用（作为输入还是输出）。并且还要根据paramTypes的值设定对应的params[x]成员的值或者是指向的地址以及缓存的长度。
   简单来说就是，先规定op是什么类型的数据，再往它的参数里写各种各样需要的数据
 */
    op.paramTypes = TEEC_PARAM_TYPES(TEEC_VALUE_INOUT, TEEC_NONE,
                     TEEC_NONE, TEEC_NONE);
    op.params[0].value.a = 42;

/*
 * TA_NEW_TAPS_CMD_INC_VALUE是要调用的TA中的实际功能。
   使用已经创建好的session，TA与CA端规定的command ID以及配置好的TEEC_Operation类型变量作为参数调用TEEC_InvokeCommand函数来真正发起请求。 调用TEEC_InvokeCommand成功之后，剩下的事情就有OP-TEE和TA进行处理并将结果和相关的数据通过TEEC_Operation类型变量中的params成员返回给CA。
   
   函数原型：TEEC_Result TEEC_InvokeCommand(TEEC_Session *session, uint32_t cmd_id,TEEC_Operation *operation, uint32_t *error_origin)
   函数作用描述：通过cmd_id和打开的session，来通知session对应的TA（因为opensession中已经指定好uuid，即规定好调用哪个TA）执行cmd_id指定的操作。
   参数说明：
session: 指向已经初始化的session结构体变量
cmd_id：TA中定义的command的ID值，让CA通知TA执行哪条command
operation: 已经初始化的TEEC_Operation类型的变量，该变量中包含CA于TA之间进行交互的buffer,缓存的属性等信息
error_origin:调用TEEC_InvokeCommand函数的时候，TA端的返回值
 */
    printf("Invoking TA to increment %d _NT\n", op.params[0].value.a);
    res = TEEC_InvokeCommand(&sess, TA_NEW_TAPS_CMD_INC_VALUE, &op,
                 &err_origin);
    if (res != TEEC_SUCCESS)
        errx(1, "TEEC_InvokeCommand failed with code 0x%x origin 0x%x",
            res, err_origin);
    printf("TA incremented value to %d _NT\n", op.params[0].value.a);

/*
	调用成功之后如果不需要再次调用该TA则需要注销session和释放掉context，这两个操作一次通过调用TEEC_CloseSession函数和TEEC_FinalizeContext函数来实现。
	closesession:关闭已经被初始化的CA与对应TA之间的session，在调用该函数之前需要保证所有的command已经执行完毕。如果session为NULL,则不执行任何操作。
	finalizecontext:释放一个已经被初始化过的类型为TEEC_Context变量，关闭CA与TEE之间的连接。在调用该函数之前必须确保打开的session已经被关闭了。
 */

    TEEC_CloseSession(&sess);

    TEEC_FinalizeContext(&ctx);

    return 0;
}
```



# 3.1 关于树莓派触发

在S盒前，设置触发，树莓派拉高引脚，示波器开始采集，通过电脑上的软件pull曲线



# 3.2 组会&超算

谷老师问题：

为什么使用这个而不使用别的同类的？为什么别的PUF关注度不高？

做这个工作的意义？既然不用这个类型的PUF为什么还要研究呢

提供了一个新的研究思路：准确度和PUF类别



上交超算文档docs.hpc.sjtu.edu.cn



# 3.2 tee_client_api&ssh传输文件



```c
/**
 * struct TEEC_Context - Represents a connection between a client application
 * and a TEE.
 */
typedef struct {
	/* Implementation defined */
	int fd; //open返回的文件描述符，usespace通过open调用kernel层的tee_device
	bool reg_mem;
	bool memref_null;
} TEEC_Context;

TEEC_Result TEEC_InitializeContext(const char *name, TEEC_Context *ctx){
    char devname[PATH_MAX] = { 0 };
	int fd = 0;
	size_t n = 0;
    
    for (n = 0; n < TEEC_MAX_DEV_SEQ; n++) {
		uint32_t gen_caps = 0;

		snprintf(devname, sizeof(devname), "/dev/tee%zu", n);
		fd = teec_open_dev(devname, name, &gen_caps);
		if (fd >= 0) {
			ctx->fd = fd;
			ctx->reg_mem = gen_caps & TEE_GEN_CAP_REG_MEM;
			ctx->memref_null = gen_caps & TEE_GEN_CAP_MEMREF_NULL;
			return TEEC_SUCCESS;
		}
	}
}


```



都要在本地终端输入



上传本地文件到服务器

```
scp /var/www/test.php root@192.168.0.101:/var/www/
```



从服务器上下载文件

```
scp root@192.168.0.101:/var/www/test.txt /var/www/local_dir
```



从服务器下载整个目录

```
scp -r root@192.168.0.101:/var/www/test /var/www/
```



上传目录到服务器

```
scp -r test root@192.168.0.101:/var/www/
```

注：把当前目录下的test目录上传到服务器的/var/www/ 目录



解压zip包

```
unzip [zip包名]
```



使用vi编辑文件

```
vi index.html
```



进入编辑模式 

```
i
```



退出编辑模式

```
ecs
```



退出并保存修改的内容

```
:wq!
```



**终于配好树莓派环境了！！**

==不能随便用sudo!!==





# 3.3 完成循环加密（c）不调用openssl接口

要求：针对trace文件，显式输出正在加密第几个明文这种。每行需要显示明文和加密后的密文。需要在树莓派上也能跑出来。问清楚什么叫在optee里运行自己的aes加密程序？



op：

aes128文件夹下(树莓派)

对每个随机生成的明文进行一次加解密操作 验证正确性 并记录对应输入输出

scp到树莓派上运行

通过接口设置触发



执行命令

```
gcc main.c aes.c -o aes -lwiringPi 
./aes
```

或者

makefile：

```
CC=gcc
CFLAGS=-Wall -O2
LIBS = -lwiringPi

aes.out: aes.o main.o
	$(CC) $(LIBS) -o $@ $^ $(CFLAGS)
	make clean

clean:
	rm -f *.o

cleanall:
	rm -f aes.out
	rm -f *.o
```

生成aes.out可执行文件，直接执行就可以了

```
./aes.out
```



# 3.3 openssl aes加密&gcc编译过程

aes128v2文件夹下，树莓派服务器上

```
#include <openssl/aes.h> 报错
虽然openssl version 显示OpenSSL 1.1.1c
但是头文件依旧报错
原因是libssl-dev～没有安装
libssl-dev包含libraries, header files and manpages，是openssl的一部分
使用sudo apt-get install libssl-dev来安装libssl-dev

 安装之后，编译不再报错，但是出现 “undefined reference to `AES_set_encrypt_key'“

应该是没有指定连接的库， 在makefile中增加 -lssl -lcrypto
```



AES-C-MASTER文件夹下（本地）

```
gcc main_test.c aes_core.c -o aes
./aes
```



openssl_aes文件夹下（树莓派上）

```
make clean
make
./this_aes
```



gcc编译过程：

预处理文件hello.i

```
gcc -E hello.c –o hello.i
```

编译成.s文件，可用vi查看

```
gcc -S hello.i
```

汇编文件.o

```
gcc -c hello.s
```

链接成可执行文件

```
gcc hello.o
```



# 3.4 makefile

以openssl_aes为例

Makefile：

```makefile

CFLAGS=-I. -I$(PWD)/include -I/usr/include


.PHONY: all
all: aes_core.o main.o
	$(CC) aes/aes_core.o main.o -o this_aes -lwiringPi
//$(CC)是gcc

aes_core.o: 
	cd aes && \
	$(CC) -c aes_core.c $(CFLAGS)

main.o:
	$(CC) -c main.c $(CFLAGS)

test:
	$(CC) aes_from_openssl.c -lcrypto -o system_openssl_aes

clean:
	rm -f aes/aes_core.o main.o system_openssl_aes this_aes
```





# 3.5 实现tiny-aes

tiny_aes_windows文件夹

tiny_aes文件夹为树莓派上运行的程序，但本地不可运行，缺包

```
cd ..(进入desktop)
scp -r pi@10.168.1.181:/home/pi/tiny_aes CPA(当前路径为desktop)
```



# 3.7 电磁信号采集

示波器2通道接树莓派7接口，接收触发，每触发一次示波器采集一次信号，在3通道输出波形

原则上可以看出aes加密的10次差不多的波形

#### 应该设置采集多少点才能刚好展示出前10次加密的波形？  

1us 一共10格 20G/S的速率采集 共200002个点 一个点1B，200002 = 200KB，故一条曲线的大小大致为198KB

#### 噪音实在太大了怎么办？

通过socket，让树莓派通知电脑采集数据，电脑再返回DONE通知树莓派开始下一次采集trace。服务器将每次trace按照trc文件保存下来。使用merge_trace.py对曲线添加头文件后并合并，然后在SCAnalyzer里面处理，先absolute再lowpass一下。



# 3.8 git&采集100000条曲线

```
git status
git log (-x) 查看最新的x
git log -p (-x) 查看最新的x的变化

git add file(.)
git commit -m "note" 添加备注
git reset --hard 版本号或前几位
git reset --hard HEAD^ 回滚至上个版本
```



采集100000条曲线的具体操作：

电磁笔竖直在cpu中心，vs里开启采集项目，树莓派运行aes，采完所有曲线之后，scp text，在handleheader/exampes.py中运行，获得merge.trs，也就是merge之后的曲线集，导入SCAnalyzer中，先absolute再lowpass，再求个correlation，获得相关曲线，如果有明显尖峰，则采集基本没问题。



# 3.9 组会发言&binascii

自己最近搞的：

来实验室的两周多，一直在做毕设，因为毕设是需要在普通环境和optee环境分别采集aes代码的，目前是普通环境下采集曲线的一整套流程全部搭好了，但是还在找比较明显的10轮的加密曲线，找好之后就可以直接进行CPA攻击密钥了。optee环境下，了解了可信应用和客户端应用在计算中的交互过程和一些具体的API，opensession(建立会话) invokecommand（要求TA进程具体执行一些命令）。

下一步首先就是要找到比较好的曲线并进行攻击。其次是要编写optee环境下的aes加密，剩下的采集曲线、攻击密钥应该就和普通环境下的流程一样了。



binascii库：

```python
import binascii

binascii.unhexlify("de8b")  
b'\xde\x8b'

binascii.hexlify(unhexlify("de8b"))
b'de8b'

a = b'worker'#先把worker转换成二进制数据然后在用十六进制表示
b'worker'

b = binascii.b2a_hex(a)
b'776f726b6572'

binascii.a2b_hex(b)
b'worker'

c = binascii.hexlify(a)#以16进制表示
b'776f726b6572'

binascii.unhexlify(c)
b'worker'

hex(88) #10进制转化成16进制
#hex只能接受整形不能接受字符串
'0x58'

1.23.hex()
'0x1.3ae147ae147aep+0'

int(0x17)
23

bin(88) #把十进制整型转换成二进制
'0b1011000'
```







# 3.10 handleheader

examples.py 给曲线添加头部并merge

```python
import os, sys
sys.path.append("/home/kreel/")

from TraceHandler import TraceHandler
from tqdm import trange
from binascii import unhexlify

def merge_file_without_header():
    filename_format = r"E:\TEE_trace\Normal\FPGA_{:0>4}.trs"#0在左边补齐称位
    handler = TraceHandler(embed_crypto_data=True)
    for idx in trange(100000, desc="processing headers", unit="file"):
        filename = filename_format.format(idx)
        handler.append_file(filename)
    #到这里 handler的filelist是100000个文件名

    number_of_points = os.path.getsize(filename_format.format(0))
    handler.set_attribute(
        NS=number_of_points,
        SC='byte',
        GT=b"Traces for Attacking TEE",
        DS=32
    )

    def xor_bytes(s1:bytes, s2:bytes):
        assert len(s1) == len(s2)
        l = len(s1)
        return bytes([s1[i] ^ s2[i] for i in range(l)])
    with open(r"E:\TEE_trace\text", "r") as fp:
        content_lines = fp.read().split('\n')

    def data_getter(cnt, i, j):
        return unhexlify(content_lines[cnt].split(' ')[0]) + unhexlify(content_lines[cnt].split(' ')[1])
    handler.generate_header()
    handler.summary()
    input()
    handler.save2trs(r"E:\TEE_trace\merge.trs", crypto_data_getter=data_getter)


def test_header():
    handler = TraceHandler(embed_crypto_data=False)
    handler.set_attribute(
        NS=int(32e6),
        SC='int8',
        GT=b"Traces for Attacking TEE",
        DS=16
    )
    return handler.generate_header_bytes()

if __name__=='__main__':
    merge_file_without_header()
```





# 3.11  修改CA TA:hky_test

https://www.bilibili.com/video/BV1L4411N7gZ?p=2

记得修改完对应的代码后还需要修改主文件夹下（optee_example）的makefile文件，加一个自己的

```
EXAMPLE_LIST := hello_word hky_test
```



```
mv source_name target_name
cp source_file ./(指当前目录下)
repo grep target_word(在当前目录下搜索想要找的字样)
find / -name *.ta (在当前目录查找目标文件)
```







