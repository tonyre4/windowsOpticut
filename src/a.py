# -*- coding: utf-8 -*-
from prettytable import PrettyTable
from subprocess import call as cmddo
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import os
import numpy as np
#from scipy.optimize import minimize,brute
from pulp import *
import pandas as pd
import sys

#sys.setdefaultencoding('utf-8')

class optiti:
    def __init__(me,dictt):
        me.mydict = dictt


class ordenes:
    def __init__(me,pers,num_prt):
        me.persianas = pers
        me.np = num_prt
        me.nps = me.np.split("-")
        me.color = me.nps[1]
        if me.nps[2]=="40":
            me.LouverLong = 192.
        else:
            me.LouverLong = 82.5

        me.ps = []
        me.slotCtr = 0

        for i,p in me.persianas.iterrows():
            me.ps.append(persiana(p,me.np))
        
        #me.computeCuts()

    def computeCuts(me):
        cuts = np.array(me.persianas.loc[:,["FinalLouverLengthNumeric", "LouverQty"]])
        

        #agregando valances
        for p in me.ps:
            if p.Vbool:
                cuts = np.vstack([cuts,[p.getVW(),1]])
            else:
                if not p.Vbool is None:
                    print("ATENCION!. El color del Valance de X no es el mismo que del louver. Este programa a√∫n no est√° preparado para esta combinacion!")
       
        #temp = []

        #for x in cuts:
        #    temp.append(float(x[0]),int(x[1]))
        #    print(x)

        

        #Valores de cortes ordenados
        twn = np.unique(cuts[:,0])
        #Obteniendo el total de cortes por cada medida
        meas = np.empty((0,2))
        for w in twn:
            s = np.sum(cuts[np.where(cuts[:,0]==w),1])
            meas = np.vstack([meas,[w,s]])


        me.computedCuts = meas
        return meas

    
    def buscarCorte2(me,corte):
        acaba = False

        for p in me.ps: #Para cada persiana
            if p.setCut(corte): #Mete el corte del argumento y lo contabiliza
                if p.slot == -1: #Si no tiene slot asignado, se le asigna con el counter
                    p.slot = int(me.slotCtr%10)+1
                    p.car = int(me.slotCtr/10)+1
                    me.slotCtr += 1
                if p.VRDY:
                    me.vals.append(True)
                    p.VRDY = False
                else:
                    me.vals.append(False)
                if p.RDY: #Si termina la orden retorna un True
                    #print("Aqui termina la orden del slot ", p.slot," del carro ", p.car)
                    acaba = True
                me.slotter.append(p.slot) #Construye el arreglo de slots
                me.carr.append(p.car) #Construye el arreglo de carros
                break
        else:
            print("Error!, No se encontro el corte")
            exit()
        return acaba

    def printdf(me,df):
        try:
            for r in df.iterrows():
                r = r[1]
                print(r.at["idx"],"\t",dict(r.at["obj"])["cortes"],"\t",r.at["slots"],"\t",r.at["carro"],"\t",r.at["term"],"\t",r.at["vals"])
            print("$$"*20)
        except:
            r = df
            print(r.at["idx"],"\t",dict(r.at["obj"])["cortes"],"\t",r.at["slots"],"\t",r.at["carro"],"\t",r.at["term"],"\t",r.at["vals"])
            print("$$"*20)
 
    def printReport3(me, optimizado,opath):

        TODO = [] # D:

        #obs = []


        #loop para asignar slots y carros para cada una de las persianas
        for o in optimizado: #Por cada patron de corte
            #oo = optiti(o) #guardarlo como un objeto
            oo = dict(o) #para buscar por nombres
            #obs.append(oo)
            for i in range(oo["veces"]): #Por el numero de veces que se tiene que hacer
                me.slotter = []
                me.carr = []
                me.vals = []
                for c in oo["cortes"]: #Por cada corte de 1 solo patron
                    me.buscarCorte2(c) #Encuentra el corte
                me.slotter = tuple(me.slotter)
                me.carr = tuple(me.carr)
                me.vals = tuple(me.vals)
                TODO.append((o,me.slotter,me.carr,me.vals))
        
        #Saca todas las formas unicas (quita las repetidas)
        unique = set(TODO)

        idxs = []
        objs = []
        sl = []
        cr = []
        v = []
        carord = []
        slotord = []
        terms = []
        vals = []
        cols = ["idx", "obj", "slots", "carro", "veces", "carord", "slotord","term","vals"]

        #Para cada patron unico se contabiliza cuantas hay
        #Y se hace un registro nuevo de todos los cortes
        for i,u in enumerate(unique):
            c = TODO.count(u) #cuantos iguales al unico hay...
            #print(u[0].mydict["cortes"], "\t\t" ,u[1],u[2], c)
            #Se obtienen los datos de cada uno
            idxs.append(i)
            objs.append(u[0])
            sl.append(u[1])
            cr.append(u[2])
            v.append(c)
            carord.append(sum(u[2])-len(u[2]))
            #slotord.append(sum(u[1]) - len(u[1]))
            slotord.append(sum([z**2 for z in u[1]]))
            t = tuple(np.zeros(len(u[1]),np.uint8).tolist())
            terms.append(t)
            vals.append(u[3])

        data = {cols[0] : idxs,
                cols[1] : objs,
                cols[2] : sl,
                cols[3] : cr,
                cols[4] : v, #veces
                cols[5] : carord,
                cols[6] : slotord,
                cols[7] : terms,
                cols[8] : vals}

        #generando un dataframe para el ordenamiento por carros y slots
        df = pd.DataFrame(data, columns = cols)
        df = df.sort_values(by=['carord','slotord'])

        #Computa el numero de carros maximos y slots maximos por carro
        
        maxc = int(me.slotCtr/10)

        maxs = []
        for i in range(maxc):
            if i != maxc:
                maxs.append(10)
        else:
            c = int(me.slotCtr%10)
            if c!= 0:
                maxs.append(c)
     

        #me.printdf(df)
        #moviendo valances al final
        for idr,r in enumerate(df.iterrows()): #por cada fila
            r = r[1]
            if any(r.loc["vals"]): #si hay algun booleano de valance
                cccf = [] #Cortes
                sssf = [] #Slots
                ccrf = [] #Carros
                vvvf = [] #Valances bools
                ccc = [] #Cortes
                sss = [] #Slots
                ccr = [] #Carros
                vvv = [] #Valances bools
                #me.printdf(r)
                dic = dict(r.loc["obj"])
                corts = list(dic["cortes"])
                for i,vv in enumerate(r.loc["vals"]): #Por cada booleano de valance
                    cort = corts[i]
                    slo = r.loc["slots"][i]
                    car = r.loc["carro"][i]
                    val = vv

                    #print(cort,slo,car,val)

                    if not vv: #Si no es un valance (osea un louver) agregar datos
                        ccc.append(cort)
                        sss.append(slo)
                        ccr.append(car)
                        vvv.append(vv)
                    else: #Si no es un valance (osea un louver) agregar datos
                        cccf.append(cort)
                        sssf.append(slo)
                        ccrf.append(car)
                        vvvf.append(vv)
                
                    #print(ccc,sss,ccr,vvv)
                    #input("$$$$")
                
                for vv in zip(cccf,sssf,ccrf,vvvf):
                    c1,s2,cr1,v1 = vv                    
                    ccc.append(c1)
                    sss.append(s2)
                    ccr.append(cr1)
                    vvv.append(v1)
        
                #print(ccc,sss,ccr,vvv)
                #input("$$$$")

                #Convirtiendo
                ccc = tuple(ccc)
                sss = tuple(sss)
                ccr = tuple(ccr)
                vvv = tuple(vvv)

                #print(ccc,sss,ccr,vvv)
                #print(df.iloc[idr])
    
                #asignando
                dic["cortes"] = ccc

                df.iat[idr,df.columns.get_loc("obj")] = frozenset(dic.items())
                df.iat[idr,df.columns.get_loc("slots")] = sss
                df.iat[idr,df.columns.get_loc("carro")] = ccr
                df.iat[idr,df.columns.get_loc("vals")] = vvv

                #print(df.iloc[idr])
                #print("$$"*20)

        #me.printdf(df)
        #print("--"*20)

        print (maxc)
        print(maxs)
        print (me.slotCtr)

        #Computa los finales de cada orden
        wo = [] # Aqui almacena de cada carro por cada slot el worktag de cada cosa
        if int(me.slotCtr%10)==0:
            addi = 0
        else:
            addi = 1

        for i in range(maxc+addi): #por cada carro
            cw = []
            print(i)
            for ii in range(maxs[i]): #Para el rango maximo de slots por carro
                #Para encontrar worktags
                for p in me.ps:
                    if p.slot == ii+1 and p.car == i+1:
                        cw.append([ii+1,p.wt])
                for iiii, sl in enumerate(df.iterrows()): #iterando cada row de cada uno
                    for iii, sc in enumerate(zip(sl[1].loc["slots"],sl[1].loc["carro"])): #por cada elemento de la tupla de slot y carro
                        s, c = sc
                        if c != i+1: #si el carro diferente continua
                            continue
                        if s == ii+1: #si el numero del slot de busqueda es el mismo
                            last = (iiii,iii) # guarda la posicion del row y posicion de la tupla
                
                tup = df.iloc[last[0]].at['term']
                tup = list(tup)
                tup [last[1]] = ii+1
                tup = tuple(tup)
                df.iat[last[0],df.columns.get_loc("term")] = tup
                
            wo.append(cw)

    
        #for r in df.iterrows():
            #print(r.loc["idx"])#,r.loc["obj"].mydict["cortes"], r.loc["slots"], r.loc["carro"], r,loc["term"])
            #exit()

        return me.writeReport(df,wo,opath) #Mete las ordenes y el resumen de cada numero de parte

    
    def calcTots(me,df):
        me.totLouv = 0
        me.totScrap = 0.
        me.totScrapP = 0.

        for r in df.iterrows():
            r = r[1]
            dic = dict(r.at["obj"])
            me.totLouv += r.loc["veces"]
            me.totScrap += dic["scrap"] * r.loc["veces"]

        me.totLouvft = me.totLouv*me.LouverLong
        me.meanScrap = me.totScrap/me.totLouv
        me.meanScrapP = (me.meanScrap/me.LouverLong)*100.


    def getTots(me,df):
        me.calcTots(df)

        ss = "Louvers totales: %d\n" %  (me.totLouv)
        ss = "Distancia total: %d ft\n" %  (me.totLouvft)
        ss += "Scrap total: %.2fin\n" % (me.totScrap)
        ss += "Scrap por Louver: %.2fin --> " % (me.meanScrap)
        ss += "%.2f%%\n" % (me.meanScrapP)
        return ss


    def writeReport(me,df,wo,opath):
        
        ss = "Numero de parte: %s\n" % (me.np)

        ss += me.getTots(df)
    
        ss += "--"*20

        ss += "\n\n\nPatrones de corte:\n"

        for i,r in enumerate(df.iterrows()):
            ss += "Patron [%d]:\n" % (i+1)
            r = r[1]
            ss += "x%d\t>>>\t[" % (r.loc["veces"])

            obj = dict(r.loc["obj"])
            corts = obj["cortes"]
            slots = r.loc["slots"]
            car = r.loc["carro"]
            term = r.loc["term"]
            vals = r.loc["vals"]


            for c in zip(corts,slots,car,term,vals):
                co,s,cr,t,v = c
                ci = int(co)
                fr = (co-ci).as_integer_ratio()
                frs = " "
                if not fr==(0,1):
                    frs += str(fr[0]) + "/" + str(fr[1]) + " "
                xx = ""
                if t >0:
                    xx += "**"
                if v:
                    ss += "V - "
                ss += "%d%s{S%d%s}{C%d},\t" % (ci,frs,s,xx,cr)

            ss = ss[:-2] + "]\t---\t[//%.2f//]\t<--\t%.2f%%\n\n" % (obj["scrap"],obj["Porcentaje"])
    
        ss += "\nSumario de ordenes:\n"

        for i,w in enumerate(wo):
            ss += "C%d -> [" % (i+1)
            for ww in w:
                ss += "S%d - {%s}, " %(ww[0],ww[1])
            ss =  ss[:-2] + "]\n"

        ss += "\n"

        nn = me.np.replace("-","_") + "_report.txt"

        with open(opath+nn,"w+") as f:
            f.write(ss)

        print(ss)

        me.df = df

        return {"LouvQty": me.totLouv,"ftLouv":me.totLouvft ,"ftScrap":me.totScrap,"meanScrap":me.meanScrap,"pMeanScrap":me.meanScrapP}



class persiana:
    def __init__(me, feats, np):
        me.feats = feats
        me.wt = feats.loc["WorkOrderNumber"]
        me.np = np
        me.nps = me.np.split("-")
        me.color = me.nps[1]
        
        me.Lqty = me.feats.loc["LouverQty"] #Cantidad de louvers
        me.Lmeas = me.feats.loc["FinalLouverLengthNumeric"] #Medida de louvers

        me.slot = -1 #Slot
        me.car = -1 #Carro

        me.Lqtymade = 0 #Lo que ya se hizo

        me.RDY = False  #si la persiana esta lista
        me.VRDY = False #Revisa si ya fue contabilizado el Valance
        me.LRDY = False #Revisa si ya estan listos los louvers
        
        if me.nps[2] == "40":
            me.LouverLong = 192.
        else:
            me.LouverLong = 82.5

        try:
            me.Vnp = feats.loc["ValanceInsert ComponentNumber"].encode("ascii", errors="ignore").decode().replace(" ","")
            me.Vnps = me.np.split("-")
            me.Vcolor = me.Vnps[1]
            if me.Vnps[2] == "40":
                me.ValanceLong = 192.
            else:
                me.ValanceLong = 82.5
            me.Vbool = me.Vcolor == me.color
            me.Vmeas = me.feats.loc["ValanceBaseWidthNumeric"]
            me.Vmade = 0
        except:
            me.Vnp = None
            me.Vbool = None
            me.Vmade = None
            me.Vmeas = None

    def setCut(me,corte): #revisa si un valance o corte fue contabilizado en la persiana
        if not me.RDY: #Si no esta lista
            aff = False #Init
            if me.Lmeas == corte and not me.LRDY: #si el corte de entrada es igual a la medida de louver
                me.Lqtymade += 1 #aumenta los cortes
                aff = True #Retorna que efectivamente se metio un corte
            else: #Si no verifica 
                if me.Vmade == 0: #Si aun  no esta contabilizado el valance
                    if me.Vmeas == corte: #si es igual a la medida del valance
                        me.Vmade = 1 #deja el valance como hecho
                        me.VRDY = True
                        aff = True #retorna que si se metiÛ un corte
            me.isReady()#Computa si la persiana ya esta lista
            return aff
        else:
            return False #Si esta lista


    def isReady(me):
        me.LRDY = me.Lqty == me.Lqtymade
        if not (me.Vmade is None):
            me.RDY = me.Vmade == 1 and me.LRDY
        else:
            me.RDY = me.LRDY
        
    
    def getVW(me):
        return me.Vmeas



class sortHandler:
    def __init__(me,stockSize,cuts,num_prt,path_r):
        me.loadsolver()

        me.num_prt = num_prt
        me.cuts = cuts.sort_values(by=["LouverQty"])
        me.ordhand = ordenes(me.cuts,me.num_prt)

        meas = me.ordhand.computeCuts()

        me.measures = meas[:,0]
        me.counters = meas[:,1]
        me.path_r = path_r
        me.orders = []
        me.ordersSimple = []
        me.ordersForReport = []
        me.stS = stockSize
        me.getOrders()
        #me.printOrders()
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', -1)
        #print(me.cuts)
        me.req = me.ordhand.printReport3(me.ordersSimple,path_r)

    def loadsolver(me): #carga el cdw.exe
        if os.name == 'nt':
            cwd = os.getcwd()
            solverdir = 'cbc.exe'  # extracted and renamed CBC solver binary
            #solverdir = 'Cbc-2.7.5-win32-cl15icl11.1\\bin\\cbc.exe'  # extracted and renamed CBC solver binary
            solverdir = os.path.join(cwd, solverdir)
            #print(solverdir)
            me.solver = pulp.COIN_CMD(path=solverdir)

    #Algorimtmo optimizador
    ########################
    ########################
    def getBest(me): #Obtiene la mejor combinacion de cortes
        prob = LpProblem("Cut_problem",LpMaximize)

        Xs = []

        for i,c in enumerate(me.counters):
            Xi=LpVariable("X{0:05d}".format(i),0,int(c),LpInteger)
            Xs.append(Xi)
            prob.objective += Xi*me.measures[i]

        c = prob.objective

        little = 0.5

        prob.objective = prob.objective == me.stS - little

        prob += me.stS - (c)  >= little , "No menor que cero"
        if os.name == 'nt':
            prob.solve(me.solver)
        else:
            prob.solve()

        XXX = np.ones(me.measures.shape[0])

        for i,v in enumerate(prob.variables()):
            #print(v.name, "=", v.varValue)
            try:
                XXX[i] = v.varValue
            except:
                pass

        return np.array(XXX,np.uint)
    
    def getOrders(me,p=False): #Funcion que computa el optimizador
        while np.sum(me.counters)>0:
            bs = me.getBest()
            bs = bs.astype(np.uint)

            div = me.counters/bs
            div = div[~np.isnan(div)]

            times = int(np.min(div))

            if p:
                print("Best:")
                print(bs)
                print("Counters:")
                print(me.counters)
                print("Iguales:", times)

            oor = []
            for n,i in enumerate(bs):
                for ii in range(i):
                    oor.append(me.measures[n])


            for i in range(times): # genera las ordenes
                me.orders.append(oor)

            scrap = me.stS-np.sum(np.array(oor))
            isScrap = (scrap>0)
            perc = scrap*100./me.stS
            percs = [x/me.stS for x in oor]

            dic = {"cortes" : tuple(oor), "veces" : times, "scrap" : scrap, "Porcentaje": perc}
            fz = frozenset(dic.items())
            
            
            me.ordersSimple.append(fz)
            #me.ordersSimple.append([oor,times,scrap,perc])
            me.ordersForReport.append([oor,percs,isScrap,times,scrap,perc])

            rest = bs*times
            #rest = rest.astype(np.uint)

            me.counters -= rest
            app = np.sum(np.array(oor))
            if p:
                print("Resta:")
                print(rest)
                print("Despues de resta:")
                print(me.counters)
                print("Aprovechado:")
                print(app)
                print("Scrap:")
                print(me.stS-app)

    ###########################################
    ###########################################
    
    ##Reportes
    def printRequ(me, p = False):
        num_tot = 0

        for i,o in enumerate(me.ordersSimple):
            o = dict(o)
            num_tot += o["veces"]
       


        me.totLouv = num_tot
        me.ftTotLouv = num_tot*me.stS
        me.req = [me.totLouv,me.ftTotLouv]
        
        if p:
            print("**Reporte de requisicion**")
            print("-"*70)
            print("Numero de parte: 68-9306-40")
            print("-"*70)
            print("Total de Louvers:")
            print("\t" + str(num_tot))
            print("Total ft:")
            print("\t" + str(me.stS*num_tot))
            print("-"*70)
            print("")

    def getReq(me): #Compila datos de requisiciÛn y scrap
        try:
            return me.req
        except Exeption as e:
            print("No se computo la requisicion")
            print(e)
            exit()
            #me.printRequ()
            #return me.req

   
    def printOrders(me,p = False, gr = True):
        me.printRequ()

        slot = 1
        carro = 1

        r = "$"*50 +"\n"
        r += "**Reporte de corrida**\n"
        r += "Numero de parte: %s\n" % me.num_prt
        r += "$"*50 +"\n"
        r += "$"*50 +"\n"

        #Inits
        prom_perc = 0.0
        prom_ft = 0.0
        num_tot = 0
        st = 0.0

        sss = ""


        for i,o in enumerate(me.ordersSimple):

            #print(o)

            #print(me.cuts)


            prom_perc += o["scrap"]*o["Porcentaje"] #acumulado de porcentajes de desperdicio
            prom_ft += o["scrap"]*o["veces"] #acumulado de desperdicio en ft
            st += o["scrap"]  #suma total de scrap

        
            ss = "Patron %d:\n" % (i+1)
            ss += "Forma del corte:\n\t"
            ss += str(o["cortes"]) + "\n"
            ss += "x%d ve" % o[1]

            if o["veces"]==1:
                ss += "z\t"
            else:
                ss += "ces\t"
            ss += "\t\tSobrante: %f = %f%%\n" % (o["scrap"], o["Porcentaje"])
            ss += "-"*70 + "\n\n"

            sss += ss

            if p:
                print(ss)


        prom_perc /= me.totLouv
        prom_ft /= me.totLouv
        me.scrapLouvers = st/me.stS
        
        me.meanScrapPercent = prom_perc
        me.meanScrapft = prom_ft
        me.sumScrapft = st

        me.req.append(me.meanScrapPercent)

        r += "Material: %d Louvers\nMaterial acumulado: %.2fin\n" % (me.totLouv,me.totLouv*me.stS)     

        #r += "Sobrante promedio (por louver): %f%%\n"%(me.meanScrapPercent)
        r += "Sobrante promedio (por louver) en pulgadas: %.2fin --> %.2f%%\n" % (me.meanScrapft, me.meanScrapPercent)

        r += "Sobrante acumulado en %d louvers: %.2fin -->  %.2f louvers\n" % (me.totLouv,me.sumScrapft,me.scrapLouvers)
        r += "$"*50 + "\n"

        r += sss

        if p:
            print(r)
        if gr:
            n_file = "%s%s_cutReport.txt" % (me.path_r, me.num_prt)
            with open(n_file,"w") as f:
                f.write(r)

        #me.pdfRun()

def printReq(path,mat): #Imprime requisicion y reporte de scrap
        x1 = PrettyTable()
        x2 = PrettyTable()

        x1.field_names = ["Numero de parte", "Distancia (ft)", "Scrap (ft)", "Scrap promedio (ft)", "Scrap (%)"]
        x2.field_names = ["Numero de parte", "Qty", "Distancia (ft)"]

        meani = []
        sss = "==Reporte de scrap Gral==\n"
        sss += str("$$"*25)
        sss += " \n"
        #sss += "Numero de parte\t\tDistancia\t\tScrap ft\t\t%Scrap\t\t\n"


        ss = "**Requisicion de materiales**\n"
        ss += str("$$"*25)
        ss += " \n"
        #ss += "Numero de parte\t\tQty\t\tTot ft\t\t%\n"
        for p in mat:
            ndp = p[0] #numero de parte

            s = ndp + "\t\t" + "%d"%(p[1]["LouvQty"]) + "\t\t" +  "%.2f"%(p[1]["ftLouv"]) + " \n"
            s1 = ndp + "\t\t" + "%.2f"%(p[1]["ftLouv"]) + "\t\t" +  "%.2f"%(p[1]["ftScrap"]) + "\t\t" +"%.2f"%(p[1]["pMeanScrap"]) + "\t\t" +"%.2f"%(p[1]["meanScrap"]) +" \n"
            #ss += s
            #sss += s1

            x1.add_row([ndp,p[1]["ftLouv"],p[1]["ftScrap"],"%.2f"%p[1]["meanScrap"],"%.2f"%p[1]["pMeanScrap"] ])
            x2.add_row([ndp,p[1]["LouvQty"],"%.2f"%p[1]["ftLouv"]])

            meani.append(p[1]["pMeanScrap"])

        ss+=x2.get_string() + "\n"
        sss+=x1.get_string() + "\n"

        #ss += "$$"*25
        sss += "$$"*25

        meani = np.array(meani)
        meani = np.mean(meani)

        sss += "\nProrcentaje de media de Scrap Global: %.2f%%" % meani

        with open(path + "requisicion.txt", "w+") as f:
            f.write(ss)
        with open(path + "scrapRprt.txt", "w+") as f:
            f.write(sss)

        #return {"LouvQty": me.totLouv,"ftScrap":me.totScrap,"meanScrap":me.meanScrap,"pMeanScrap":me.meanScrapP}

def browsefunc(ent,mode):
    if mode==0:
        filename = filedialog.askopenfilename()
    if mode==1:
        filename = filedialog.askdirectory() + "/"
    
    ent.delete(0,tk.END)
    ent.insert(0,filename)
    
def checker(e1,e2,root):
    print("entra")
    R = dosolve(e1.get(),e2.get(),root)
    if R=="opth":
        messagebox.showerror("Error","No se selecciono una carpeta de salida valida.")
    if R=="noread":
        messagebox.showerror("Error","No se selecciono un archivo csv valido.")
    if R=="csv":
        messagebox.showerror("Error","No se selecciono un archivo csv.")
    if R=="OK":
        messagebox.showinfo("OK", "Reporte generado con exito")
        cmddo("explorer.exe %s"%e2.get().replace("/","\\"),shell=True)

def g(w,c,r):
    w.grid(column=c,row=r)

#if __name__ == "__main__":
def main():
    root = tk.Tk()
    #Titulo
    root.title("DEMO - Optimizador de cortes")
    #root.iconbitmap("dmm.ico")
    root.resizable(False, False)
    
    #Widgets
    entrycsv = ttk.Entry(root,width=100)
    excsv = tk.Button(root,text="...",command= lambda: browsefunc(entrycsv,0))
    pathsal = ttk.Entry(root,width=100)
    pathsalb = tk.Button(root,text="...",command= lambda: browsefunc(pathsal,1))
    bsol = tk.Button(root,text="Resolver",command=lambda:checker(entrycsv,pathsal, root))

    #Packing
    g(tk.Label(root, text="Elige archivo csv:"),0,0)
    g(entrycsv,0,1)
    g(excsv,1,1)
    g(tk.Label(root, text="Elige carpeta de destino:"),0,2)
    g(pathsal,0,3)
    g(pathsalb,1,3)
    g(bsol,1,4)

    messagebox.showinfo("Atencion!", "Esto es una version Beta por lo que no se garantiza un funcionamiento optimo")
    root.mainloop()


def dosolve(csv,opth, root):
    np.seterr(divide='ignore')

    path_r = "./reports/"

    type_dict  = {"WorkOrderNumber" : "str", "LouverComponentNumber": "str", "FinalLouverLengthNumeric":"float", "LouverQty" : "int"}

    if opth=="":
        print("opth: ",opth)
        return "opth"
    if csv=="":
        return "csv"

    try:
        data = pd.read_csv(csv,sep=",",dtype = type_dict)
    except Exception as e:
        print(e)
        return "noread"
        
    
    #Solo louvers
    sl = data.dropna(axis=0, subset=['FinalLouverLengthNumeric'])
    
    #color,track_width,qty_cuts
    din = sl.loc[:,["WorkOrderNumber","LouverComponentNumber","FinalLouverLengthNumeric","LouverQty","ValanceInsert ComponentNumber","ValanceBaseWidthNumeric"]].copy()
   
    #obtener numeros de parte
    req = [] #lista de requisici√≥n
    parts = din.LouverComponentNumber.unique()

    ##para cada numero de parte
    lll = len(parts)
    reqs = []
    for i,pp in enumerate(parts):
   
        p = pp.encode("ascii", errors="ignore").decode().replace(" ","")

        print("Computando solucion para numero de parte %s\n%d de %d" % (p,i+1,lll))

        cuts = din.loc[din["LouverComponentNumber"]== pp ,["WorkOrderNumber","FinalLouverLengthNumeric","LouverQty", "ValanceInsert ComponentNumber","ValanceBaseWidthNumeric"]]

        #Corre optimizador
        A = sortHandler(192.,cuts,p,opth)
        reqs.append([p,A.getReq()])
    
    printReq(opth,reqs)
        #exit()
    return "OK"

    
