https://www.bilibili.com/video/BV1db4y1d79C

克隆仓库 `git clone <git地址>`
		初始化仓库：`git init` 

添加文件到暂存区：`git add -A`
		把暂存区的文件提交到仓库：`git commit -m "提交信息"`
		查看提交的历史记录：`git log --stat`

工作区回滚：`git checkout <filename>`
		撤销最后一次提交：`git reset HEAD^1`

以当前分支为基础新建分支：`git checkout -b <branchname>`
		列举所有的分支：`git branch`
		单纯地切换到某个分支：`git checkout <branchname>`
		删掉特定的分支：`git branch -D <branchname>`
		合并分支：`git merge <branchname>`



本地同步远程仓库 `git remote add origin <https://.....git>`

重命名 `git branch -M main`

拉取远程 `		git pull --rebase origin main`  （如果本地还没commit可以先 `git pull`）

再push `git push -u origin main`



## Q1:

一开始git push的时候报错 error: failed to push some refs to 'https://github.com/...

原因是远程库与本地库不一致造成的

方法：

需要先`git pull --rebase origin main` 将刚刚commit的内容保存为补丁

再``git push -u origin main``



## Q2:

pull或push的时候报错 [OpenSSL](https://so.csdn.net/so/search?q=OpenSSL&spm=1001.2101.3001.7020) SSL_read: Connection was reset, errno 10054

方法：

更改网络认证设置 `git config --global http.sslVerify "false"`

增加缓冲 `git config http.postBuffer 524288000`





## Q3：

push的时候报错  Failed to connect to github.com port 443: Timed out

方法：

先在github里添加自己的公钥：在命令行中clip id_rsa.pub，然后复制到github中

然后确认git配置好公钥：

找到git\etc\ssh\ssh_config文件 在最后追加

```
Host github.com

User git

Hostname ssh.github.com

PreferredAuthentications publickey

IdentityFile ~/.ssh/id_ed25519

Port 443

其中的 IdentityFile ~/.ssh/id_ed25519 需要换成自己的公钥路径；做完以上步骤后就可以用git bash更新代码了；
```



# Q4 上面两种都不管用了？？？

焯 好像是网的问题 捏麻的



# Q5

10054或443 还尝试了以下的办法

控制面板-网络和Internet-Internet选项-连接-局域网设置-自动检测设置我给勾上了。

