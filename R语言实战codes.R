help("function")                                     #查看函数功能
vignette("function")                                 #返回的vignette文档是 PDF的实用介绍性文章

rm(objectlist)
savehistory("name")                                  #保存历史命令到文件
loadhistory("name")                                  #载入历史命令文件
save.image("name")                                   #保存工作空间
save(objectlist, file = "name")                      #保存指定对象到文件
load("myfile")                                       #读取工作空间到当前会话
q()

options()                                            #显示设置
options(digits = 3)                                  #修改设置，显示小数点后三位

source("filename")                                   #可在当前会话中执行一个脚本

length(object)                                       #显示对象中元素/成分的数量
dim(object)                                          #显示某个对象的维度
str(object)                                          #显示某个对象的结构
class(object)                                        #显示某个对象的类或类型
mode(object)                                         #显示某个对象的模
names(object)                                        #显示某对象中各成分的名称
c(object, object,…)                                  #将对象合并入一个向量

   