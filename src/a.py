# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
#from scipy.optimize import minimize,brute
from pulp import *
import pandas as pd

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
        cuts = np.array(me.persianas.loc[:,["TrackWidthNumeric", "LouverQty"]])
        

        #agregando valances
        for p in me.ps:
            if p.Vbool:
                cuts = np.vstack([cuts,[p.getVW(),1]])
            else:
                if not p.Vbool is None:
                    print("ATENCIÓN!. El color del Valance de X no es el mismo que del louver. Este programa aún no está preparado para esta combinación!")
        

        #Valores de cortes ordenados
        twn = np.unique(cuts[:,0])
        #Obteniendo el total de cortes por cada medida
        meas = np.empty((0,2))
        for w in twn:
            s = np.sum(cuts[np.where(cuts[:,0]==w),1])
            meas = np.vstack([meas,[w,s]])

        #print("Cuts:")
        #print(cuts)
        #print("Unique")
        #print(twn)
        #print("Sum:")
        #print(meas)

        me.computedCuts = meas
        return meas

    def buscarCorte(me,corte):
        acaba = False

        for p in me.ps:
            if p.setCut(corte):
                if p.slot == -1:
                    p.slot = int(me.slotCtr%10)+1
                    p.car = int(me.slotCtr/10)+1
                    me.slotCtr += 1
                if p.RDY:
                    #print("Aqui termina la orden del slot ", p.slot," del carro ", p.car)
                    acaba = True
                me.slotternow.append(p.slot)
                break
        else:
            print("Error!, No se encontró el corte")
            exit()
        #print("Corte:",corte)
        #print("Slot:", p.slot)
        #print("Carro:", p.car)
        return acaba
    
    def buscarCorte2(me,corte):
        acaba = False

        for p in me.ps:
            if p.setCut(corte):
                if p.slot == -1:
                    p.slot = int(me.slotCtr%10)+1
                    p.car = int(me.slotCtr/10)+1
                    me.slotCtr += 1
                if p.RDY:
                    #print("Aqui termina la orden del slot ", p.slot," del carro ", p.car)
                    acaba = True
                me.slotter.append(p.slot)
                me.carr.append(p.car)
                break
        else:
            print("Error!, No se encontró el corte")
            exit()
        #print("Corte:",corte)
        #print("Slot:", p.slot)
        #print("Carro:", p.car)
        return acaba

    def printReport2(me, optimizado):
        opti = optimizado.copy()

        for ip,p in enumerate(me.ps):
            print("Computando orden %d" % ip)
            while not p.RDY:
                ##Encontrar el mejor de todos los cortes que puede llenar el pedido
                maxoc = 0
                maxind = None
                for ii,o in enumerate(opti):
                    if o["veces"]==0:
                        continue
                    if p.Lmeas in o["cortes"]:
                        cnt = o["cortes"].count(p.Lmeas)
                        if maxoc<cnt:
                            maxind = ii
                ##Ya encontrado ahora hay que ir bajando las veces en cada corte
                try:
                    mc = opti[maxind]
                except:
                    print(p.Lqty,p.Lqtymade)
                    exit()

                for i in range(mc["veces"]):
                    for c in mc["cortes"]:
                        if c == p.Lmeas: #Si es un corte de la persiana
                            p.Lqtymade += 1
                            p.isReady()
                        elif c == p.Vmeas and p.Vmade == 0: ## Por si encuentra el valance
                            p.Vmade = 1
                            p.isReady()
                        else: # Buscar de quien es
                            for ii,pp in enumerate(me.ps):
                                if ii==ip:
                                    continue
                                if pp.Lmeas == c:
                                    p.Lqtymade += 1
                                    pp.isReady()
                                elif c == pp.Vmeas and pp.Vmade == 0:
                                    pp.Vmade = 1
                                    p.isReady()
                    mc["veces"] -= 1
                    if p.RDY:
                        break

    def printReport3(me, optimizado):

        TODO = [] # D:

        obs = []

        for o in optimizado:
            oo = optiti(o)
            obs.append(oo)
            for i in range(o["veces"]):
                me.slotter = []
                me.carr = []
                for c in o["cortes"]:
                    me.buscarCorte2(c)
                me.slotter = tuple(me.slotter)
                me.carr = tuple(me.carr)
                TODO.append((oo,me.slotter,me.carr))
        
        unique = set(TODO)

        idxs = []
        objs = []
        sl = []
        cr = []
        v = []
        carord = []
        slotord = []
        terms = []

        cols = ["idx", "obj", "slots", "carro", "veces", "carord", "slotord","term"]

        for i,u in enumerate(unique):
            c = TODO.count(u)
            #print(u[0].mydict["cortes"], "\t\t" ,u[1],u[2], c)
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

        data = {cols[0] : idxs,
                cols[1] : objs,
                cols[2] : sl,
                cols[3] : cr,
                cols[4] : v,
                cols[5] : carord,
                cols[6] : slotord,
                cols[7] : terms}

        df = pd.DataFrame(data, columns = cols)


        df = df.sort_values(by=['carord','slotord'])

        maxc = int(me.slotCtr/10)
        maxs = []
        for i in range(maxc):
            if i != maxc:
                maxs.append(10)
        else:
            c = int(me.slotCtr%10)
            if c!= 0:
                maxs.append(int(me.slotCtr%10))
      
        wo = []
 
        for i in range(maxc+1): #por cada carro
            cw = []
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
                
                #print("##"*20)
                #tup = slots.iloc[last[0]].at['term'][last[1]] = ii+1
                tup = df.iloc[last[0]].at['term']
                tup = list(tup)
                #print(tup)
                tup [last[1]] = ii+1
                tup = tuple(tup)
                #print(df)
                df.iat[last[0],df.columns.get_loc("term")] = tup
                #print(df)
                #print("##"*20)
                
            wo.append(cw)
                #print(slots)
                #exit()
                

        me.writeReport(df,wo)


        #Orden por carros


    def writeReport(me,df,wo):
        
        ss = "Patrones de corte:\n"

        for r in df.iterrows():
            r = r[1]
            ss += "%d veces:\t[" % (r.loc["veces"])

            obj = r.loc["obj"].mydict
            corts = obj["cortes"]
            slots = r.loc["slots"]
            car = r.loc["carro"]
            term = r.loc["term"]

            for c in zip(corts,slots,car,term):
                co,s,cr,t = c
                if t >0:
                    xx = "**"
                else:
                    xx = ""
                ss += "%.2f {S%d%s}{C%d},\t" % (co,s,xx,cr)
            
            ss = ss[:-2] + "]\t---\t[//%.2f//]\t<--\t%.2f%%\n\n" % (obj["scrap"],obj["Porcentaje"])
    
        ss += "Sumario de ordenes:\n"

        for i,w in enumerate(wo):
            ss += "C%d -> [" % (i+1)
            for ww in w:
                ss += "S%d - {%s}, " %(ww[0],ww[1])
            ss =  ss[:-2] + "]\n"

        ss += "\n"
            #print(ss)
            #print(obj)
            #exit()


        print(ss)

        me.df = df



    def printReport(me, optimizado):
        me.slotterpast = []
        me.slotternow = []
        me.terminados = []
        me.veces = 0

        #print(me.cuts)

        for o in optimizado:
            me.o = o
            #print("Para el corte:", o["cortes"])
            me.veces = 0
            me.slotterpast = []
            for i in range(o["veces"]):
                impreso = False
                me.terminados = []
                me.slotternow = []
                for c in o["cortes"]:
                    me.terminados.append(me.buscarCorte(c))

                if me.slotterpast != []:
                    if me.slotterpast == me.slotternow:
                        me.veces += 1
                    else:
                        if me.veces != 0:
                            me.printForma(o["cortes"], me.veces, me.slotterpast, None)
                        impreso = True
                        me.veces = 1

                    if any(me.terminados) and not impreso:
                        if me.veces != 0:
                            me.printForma(o["cortes"], me.veces, me.slotternow, None)
                        impreso = True
                        me.veces = 0
                else:
                    me.veces += 1

                me.slotterpast = me.slotternow.copy()
            else:
                if not impreso:
                    if me.veces != 0:
                        me.printForma(o["cortes"], me.veces, me.slotternow, None)

    def printForma(me, patron, veces, slots, carro):
        ss =  "Patron:\t" + str(patron) + " --- [//%.2f//] -> %.2f%%\n" % (me.o["scrap"], me.o["Porcentaje"])
        ss += "Slots: \t["
        for i, s in enumerate(slots):
            if me.terminados[i]:
                z = "* "
            else:
                z = " "
            ss += str(s) + z
        ss = ss[:-1] + "]\n"
        ss += "Carro:" + "\n"
        ss += "Veces:  x%d" % veces + "\n\n"

        print(ss)



class persiana:
    def __init__(me, feats, np):
        me.feats = feats
        me.wt = feats.loc["WorkOrderNumber"]
        me.np = np
        me.nps = me.np.split("-")
        me.color = me.nps[1]
        
        me.Lqty = me.feats.loc["LouverQty"] #Cantidad de louvers
        me.Lmeas = me.feats.loc["TrackWidthNumeric"] #Medida de louvers

        me.slot = -1 #Slot
        me.car = -1 #Carro

        me.Lqtymade = 0 #Lo que ya se hizo

        me.RDY = False  #si la persiana esta lista

        
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

    def setCut(me,corte):
        if not me.RDY:
            aff = False
            if me.Lmeas == corte:
                me.Lqtymade += 1
                aff = True
            else:
                if me.Vmade == 0:
                    if me.Vmeas == corte:
                        me.Vmade = 1
                        aff = True

            me.isReady()
            return aff
        else:
            return False


    def isReady(me):
        if not (me.Vmade is None):
            me.RDY = me.Vmade == 1 and me.Lqty == me.Lqtymade
        else:
            me.RDY = me.Lqty == me.Lqtymade
    
    def getVW(me):
        return me.Vmeas



class sortHandler:
    def __init__(me,stockSize,cuts,num_prt,path_r):
        me.load()

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
        print(me.cuts)
        me.ordhand.printReport3(me.ordersSimple)

    def load(me):
        cwd = os.getcwd()
        solverdir = 'cbc.exe'  # extracted and renamed CBC solver binary
        #solverdir = 'Cbc-2.7.5-win32-cl15icl11.1\\bin\\cbc.exe'  # extracted and renamed CBC solver binary
        solverdir = os.path.join(cwd, solverdir)
        print(solverdir)
        me.solver = pulp.COIN_CMD(path=solverdir)

    #Algorimtmo optimizador
    ########################
    ########################
    def getBest(me):
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
        #prob.solve()
        prob.solve(me.solver)


        XXX = np.ones(me.measures.shape[0])

        for i,v in enumerate(prob.variables()):
            #print(v.name, "=", v.varValue)
            try:
                XXX[i] = v.varValue
            except:
                pass

        return np.array(XXX,np.uint)
    
    def getOrders(me,p=False):
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

            
            me.ordersSimple.append({"cortes" : oor, "veces" : times, "scrap" : scrap, "Porcentaje": perc})
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

    def getReq(me):
        try:
            return me.req
        except:
            me.printRequ()
            return me.req

   
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




def printReq(mat):
        ss = "**Requisicion de materiales**\n"
        ss += str("$$"*25)
        ss += " \n"
        ss += "Numero de parte\t\tQty\t\tTot ft\t\t%\n"
        for p in mat:
            s = p[0] + "\t\t" + "%d"%(p[1][0]) + "\t\t" +  "%.2f"%(p[1][1]) + "\t\t" + "%.2f%%"%(p[1][2]) + " \n"
            ss += s

        ss += "$$"*25
        with open("req.txt", "w+") as f:
            f.write(ss)

    


#if __name__ == "__main__":
def main():
    np.seterr(divide='ignore')
    try:
        csv = sys.argv[1]
    except:
        print("No hay archivo csv de entrada...")
        exit()

    path_r = "./reports/"

    data = pd.read_csv(csv,sep=";")
    
    
    #Solo louvers
    sl = data.dropna(axis=0, subset=['TrackWidthNumeric'])
    
    #color,track_width,qty_cuts
    din = sl.loc[:,["WorkOrderNumber","LouverComponentNumber","TrackWidthNumeric","LouverQty","ValanceInsert ComponentNumber","ValanceBaseWidthNumeric"]].copy()
   
    #obtener numeros de parte
    req = [] #lista de requisición
    parts = din.LouverComponentNumber.unique()

    ##para cada numero de parte
    lll = len(parts)
    reqs = []
    for i,pp in enumerate(parts):
   
        p = pp.encode("ascii", errors="ignore").decode().replace(" ","")

        print("Computando solución para numero de parte %s\n%d de %d" % (p,i+1,lll))

        cuts = din.loc[din["LouverComponentNumber"]== pp ,["WorkOrderNumber","TrackWidthNumeric","LouverQty", "ValanceInsert ComponentNumber","ValanceBaseWidthNumeric"]]

        #Corre optimizador
        A = sortHandler(192.,cuts,p,path_r)
        reqs.append([p,A.getReq()])
    
    #printReq(reqs)
        #exit()

    
