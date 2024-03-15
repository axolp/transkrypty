import jieba

text= "因为爱这个事本来是一种自友的感情"

#jieba.enable_paddle()
odp= list(jieba.cut(text))

print(odp)