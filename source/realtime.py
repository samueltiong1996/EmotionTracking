from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
    import ttk
else:
    import tkinter as Tk
    from tkinter import ttk
with sqlite3.connect('database.db') as db:
    c=db.cursor()

root = Tk.Tk()
root.wm_title("Real-time Graph")
root.geometry('800x500')

productc =[]

c.execute('SELECT DISTINCT PRODUCT_CODE FROM PRODUCT')
for row in c.fetchall():
    productc.append(row[0])

cb1 = ttk.Combobox(master=root,state="readonly",values=productc)
cb1.pack(side=Tk.TOP)

fig = plt.figure(figsize=(7,4))
plt.suptitle('Real-time Graph on Percentage of Emotion Against Time', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Percentage (%)')
ax1 = fig.add_subplot(1,1,1)

canvas = FigureCanvasTkAgg(fig,master = root)
canvas.draw()
canvas.get_tk_widget().pack(side=Tk.BOTTOM,fill=Tk.BOTH,expand=True)


#animated graph which run every 1 second to make sure changes in database make an instant update on graph
def animate(i):
    
    ax1.clear()
    total = 0
    linechoice = 'SELECT * FROM TEMP WHERE PRODUCT_CODE = ?'
    c.execute(linechoice,[cb1.get()])
    ids =[]
    timest = []
    ehappy = []
    esad = []
    eangry = []
    esuprised = []
    enormal = []


    for row in c.fetchall():
        total = row[2]+row[3]+row[4]+row[5]+row[6]
        ids.append(row[0])
        timest.append(row[1])
        #ehappy.append(((float(row[2]) / float(total) * 100.00)))
        #esad.append(((float(row[3]) / float(total) * 100.00)))
        #eangry.append(((float(row[4]) / float(total) * 100.00)))
        #esuprised.append(((float(row[5]) / float(total) * 100.00)))
        #enormal.append(((float(row[6]) / float(total) * 100.00)))

        ehappy.append(row[2])
        esad.append(row[3])
        eangry.append(row[4])
        esuprised.append(row[5])
        enormal.append(row[6])

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Emotions") 
        
    d1 = ax1.plot(timest,ehappy,color="green",label="Happy")
    d2 = ax1.plot(timest,esad,color="blue",label="Sad")
    d3 = ax1.plot(timest,eangry,color="red",label="Angry")
    d4 = ax1.plot(timest,esuprised,color="purple",label="Suprised")
    d5 = ax1.plot(timest,enormal,color="grey",label="Normal")
    plt.ylabel('Emotion Count')
    plt.legend(bbox_to_anchor=(1,1),loc=1,borderaxespad=0)    

ani = animation.FuncAnimation(fig,animate,interval=1000)
root.mainloop()