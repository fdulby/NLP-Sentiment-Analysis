# Git 终端操作入门指南

这份文档面向刚开始使用 Git 的同学，目标是让你能在终端里安全地管理本项目文件：查看改动、提交需要的文件、推送到 GitHub，以及正确创建、删除、移动文件。

本项目仓库地址：

```bash
https://github.com/fdulby/NLP-Sentiment-Analysis
```

当前项目目录示例：

```bash
/Users/liubingyi/Documents/project/NLP-Sentiment-Analysis/PJ8_LSTM
```

## 1. Git 的基本概念

Git 可以理解为项目的版本管理工具。你每次完成一部分修改后，可以把这些修改保存成一个“版本快照”。

常见流程是：

```text
修改文件 -> 查看状态 -> 添加到暂存区 -> 提交 commit -> 推送 push 到 GitHub
```

对应命令是：

```bash
git status
git add 文件名
git commit -m "提交说明"
git push
```

## 2. 进入项目目录

打开终端后，先进入项目目录：

```bash
cd /Users/liubingyi/Documents/project/NLP-Sentiment-Analysis/PJ8_LSTM
```

确认自己在正确位置：

```bash
pwd
```

查看当前目录下有哪些文件：

```bash
ls
```

## 3. 查看 Git 状态

每次操作前后都建议运行：

```bash
git status
```

你会看到几类信息：

- `modified`：文件被修改了。
- `new file` 或 `untracked files`：新文件，还没有被 Git 跟踪。
- `deleted`：文件被删除了。
- `nothing to commit`：当前没有需要提交的改动。

## 4. 只提交你需要的文件

不要一上来就盲目使用：

```bash
git add .
```

它会把当前目录下所有改动都加入提交，容易把不需要的文件也提交上去，例如模型文件、临时输出、缓存文件。

更安全的做法是指定文件：

```bash
git add train.py
git add predict.py
git add git.md
```

也可以一次添加多个文件：

```bash
git add train.py predict.py git.md
```

添加后再检查：

```bash
git status
```

如果确认无误，再提交：

```bash
git commit -m "补充 Git 使用说明"
```

推送到 GitHub：

```bash
git push
```

## 5. 查看文件具体改了什么

查看所有未暂存的修改：

```bash
git diff
```

查看某个文件改了什么：

```bash
git diff train.py
```

查看已经 `git add` 之后、准备提交的内容：

```bash
git diff --staged
```

这个命令很重要。提交前最好看一遍，确认没有把错误内容提交进去。

## 6. 创建文件和文件夹

创建普通文件：

```bash
touch notes.md
```

创建文件夹：

```bash
mkdir docs
```

创建多层文件夹：

```bash
mkdir -p docs/images
```

注意：按照本项目约定，每次创建文件或文件夹前，应该先提醒项目负责人。

创建后查看状态：

```bash
git status
```

让 Git 跟踪新文件：

```bash
git add notes.md
```

## 7. 删除文件和文件夹

删除文件前，先确认文件名：

```bash
ls
```

删除普通文件：

```bash
rm notes.md
```

如果这个文件已经被 Git 跟踪，推荐使用：

```bash
git rm notes.md
```

这样 Git 会同时记录“文件被删除”这个操作。

删除空文件夹：

```bash
rmdir docs
```

删除非空文件夹要非常小心：

```bash
rm -r docs
```

不要随便使用：

```bash
rm -rf 文件夹名
```

`rm -rf` 很危险，删错后通常很难恢复。使用前一定要确认路径。

## 8. 移动或重命名文件

普通移动文件：

```bash
mv old_name.py new_name.py
```

如果文件已经被 Git 跟踪，推荐使用：

```bash
git mv old_name.py new_name.py
```

移动文件到文件夹：

```bash
git mv train.py src/train.py
```

移动后检查：

```bash
git status
```

## 9. 撤销还没提交的修改

如果你修改了某个文件，但还没有 `git add`，想撤销：

```bash
git restore train.py
```

如果你已经 `git add train.py`，想从暂存区拿出来，但保留文件内容：

```bash
git restore --staged train.py
```

如果你已经提交了，就不要随便撤销，先查看历史：

```bash
git log --oneline
```

不确定时，先不要执行 `reset`、`rebase`、`checkout` 这类命令。

## 10. 拉取远程最新代码

开始修改前，建议先拉取 GitHub 上的最新代码：

```bash
git pull
```

如果提示冲突，不要慌，先运行：

```bash
git status
```

然后根据提示查看哪些文件冲突。冲突解决完后再：

```bash
git add 冲突文件
git commit -m "解决合并冲突"
git push
```

## 11. 推荐的日常工作流程

每次准备改项目时：

```bash
cd /Users/liubingyi/Documents/project/NLP-Sentiment-Analysis/PJ8_LSTM
git pull
git status
```

修改文件后：

```bash
git status
git diff
```

只添加需要提交的文件：

```bash
git add 文件1 文件2
```

检查准备提交的内容：

```bash
git diff --staged
```

提交：

```bash
git commit -m "简短说明这次改了什么"
```

推送：

```bash
git push
```

## 12. 本项目里哪些文件要谨慎提交

机器学习项目通常会产生很多大文件或结果文件，例如：

- `runs/` 下的模型、图片、预测结果。
- `best_model.pt` 这类模型权重文件。
- 临时测试文件。
- Python 缓存文件，例如 `__pycache__/`。

提交前一定要用：

```bash
git status
git diff --staged
```

确认只提交真正需要进入仓库的文件。

如果某些文件永远不希望提交，可以放进 `.gitignore`。例如：

```gitignore
__pycache__/
*.pyc
.DS_Store
runs/
```

是否忽略 `runs/` 要看项目要求：如果实验结果需要提交，就不要忽略；如果只是本地训练输出，就应该忽略。

## 13. 常见问题

### 忘记自己改了什么

```bash
git status
git diff
```

### 添加错文件到暂存区

```bash
git restore --staged 文件名
```

### 想撤销某个文件的本地修改

```bash
git restore 文件名
```

### 提交说明写什么

写清楚这次改动的目的即可，例如：

```bash
git commit -m "补充数据预处理说明"
git commit -m "修复预测脚本路径问题"
git commit -m "新增实验结果可视化"
```

### 推送失败怎么办

先拉取远程更新：

```bash
git pull
```

如果没有冲突，再推送：

```bash
git push
```

如果出现冲突，先不要乱删文件，运行：

```bash
git status
```

看清楚哪些文件需要处理。

## 14. 最重要的安全原则

1. 修改前先 `git status`。
2. 提交前先 `git diff --staged`。
3. 尽量用 `git add 文件名`，不要随便 `git add .`。
4. 删除文件前先 `ls` 确认路径。
5. 不确定时不要用 `rm -rf`、`git reset --hard`。
6. 每次提交只做一类事情，不要把很多无关修改混在一起。

