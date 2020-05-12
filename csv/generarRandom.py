import random

nr = 10

mini = 50
ran = random.randint(0,50)
tot = mini + ran

l = []

with open("limpio.csv","r") as f:
    for ll in f:
        l.append(ll)
heads = l[0]
l = l[1:]

for i in range(nr):
    random.shuffle(l)
    ff = l[:tot]
    with open("Prueba"+str(i)+".csv" , "w+") as f:
        f.write(heads)
        for lll in ff:
            f.write(lll)




