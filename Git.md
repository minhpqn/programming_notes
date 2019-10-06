# Git Notes

Những ghi chép cá nhân khi sử dụng và học cách sử dụng Git cho các projects

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







