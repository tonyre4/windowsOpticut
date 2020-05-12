import pandas as pd

csv = "todos.csv"
type_dict  = {"WorkOrderNumber" : "str","LouverComponentNumber":"str"}
data = pd.read_csv(csv,sep=",",dtype = type_dict)


#data = data.drop(data[data["LouverComponentNumber"].count("-")<2].index)

ddd = pd.DataFrame(columns=data.keys())

#print(ddd)

#exit()

for d in data.iterrows():
    i = d[0]
    d = d[1]
    if not str(d["LouverComponentNumber"]).count("-")<2:
        #print(d["LouverComponentNumber"])
        ddd = ddd.append(d)
    #else:
        #print("Hay uno asi")
        #print(d["LouverComponentNumber"])
        #exit()

ddd.to_csv('limpio.csv')
