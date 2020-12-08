# Git Notes

Những ghi chép cá nhân khi sử dụng và học cách sử dụng Git cho các projects


## Strikethrough trong Markdown

Sử dụng `~~ ~~`

Tham khảo: [Basic writing and formatting syntax](https://docs.github.com/en/free-pro-team@latest/github/writing-on-github/basic-writing-and-formatting-syntax)

#### Autoreload

Thêm 2 dòng dưới đây vào notebook khi muốn notebook tự động load những thay đổi trong các file.

```
%reload_ext autoreload
%autoreload 2
```

#### How to Write a Git Commit Message

Tham khảo: [https://chris.beams.io/posts/git-commit/](https://chris.beams.io/posts/git-commit/)

#### Merge dev branch into master

```
I generally like to merge master into the development first so that if there are any conflicts, I can resolve in the development branch itself and my master remains clean.

(on branch development)$ git merge master
(resolve any merge conflicts if there are any)
git checkout master
git merge development (there won't be any conflicts now)

There isn't much of a difference in the two approaches, but I have noticed sometimes that I don't want to merge the branch into master yet, after merging them, or that there is still more work to be done before these can be merged, so I tend to leave master untouched until final stuff.

EDIT: From comments

If you want to keep track of who did the merge and when, you can use --no-ff flag while merging to do so. This is generally useful only when merging development into the master (last step), because you might need to merge master into development (first step) multiple times in your workflow, and creating a commit node for these might not be very useful.

git merge --no-ff development
```

Tham khảo: [https://stackoverflow.com/questions/14168677/merge-development-branch-with-master](https://stackoverflow.com/questions/14168677/merge-development-branch-with-master)

#### Overwrite a branch with another branch

```
# overwrite master with contents of seotweaks branch (seotweaks > master)
git checkout seotweaks    # source name
git merge -s ours master  # target name
git checkout master       # target name
git merge seotweaks # source name
```

Reference: [https://gist.github.com/ummahusla/8ccfdae6fbbe50171d77](https://gist.github.com/ummahusla/8ccfdae6fbbe50171d77)

#### Set tracking information for a branch

```
git branch --set-upstream-to=origin/dev dev
```

#### Merge another branch into your active branch

```
git merge <branch>
```

#### Xoá files khỏi git

```
git rm --cached mylogfile.log
```

For a directory:

```
git rm --cached -r mydirectory
```

#### Branching trong git

Tạo branch trong git

```
git checkout -b <branch_name>
```

Chuyển sang master

```
git checkout master
```

Xoá branch:

```
git branch -d feature_x
```

git - the simple guide: [http://rogerdudler.github.io/git-guide/](http://rogerdudler.github.io/git-guide/)


#### Determine the URL that a local Git repository was originally cloned from

```
git config --get remote.origin.url
git remote show origin
```

#### Thay đổi remote's URL

Sử dụng lệnh:

    git remote set-url origin https://github.com/USERNAME/OTHERREPOSITORY.git

Tham khảo tại: [Changing a remote's URL](https://help.github.com/articles/changing-a-remote-s-url/)

#### Thay đổi remote URLs từ HTTPS sang SSH

Sử dụng lệnh:

    git remote set-url origin git@github.com:USERNAME/OTHERREPOSITORY.git

Tham khảo tại: [Changing a remote's URL](https://help.github.com/articles/changing-a-remote-s-url/)

#### Check status của bitbucket.org

Vào trang: [http://status.bitbucket.org](http://status.bitbucket.org)

#### Tạo SSH keys

Tham khảo trang [Generating SSH keys](https://help.github.com/articles/generating-ssh-keys/)

#### Fork a repo và sync với upstream

Dùng lệnh:

    git remote add upstream <tên của repo trên upstream>

Tham khảo tại: [https://help.github.com/articles/fork-a-repo/](https://help.github.com/articles/fork-a-repo/).







