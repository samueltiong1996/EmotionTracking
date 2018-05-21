import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sqlite3
with sqlite3.connect('database.db') as db:
    c=db.cursor()


fig = plt.figure(figsize=(7,4))
plt.suptitle('Real-time Graph on Percentage of Emotion Against Time', fontsize=14)
plt.xlabel('Time')
plt.ylabel('Percentage (%)')
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    

    c.execute('SELECT rowid, times, happy, sad FROM graph')
    ids =[]
    times =[]
    emotion1 = []
    emotion2 = []

    for row in c.fetchall():
        ids.append(row[0])
        times.append(row[1])
        emotion1.append(row[2])
        emotion2.append(row[3])

    ax1.clear()            
    ax1.plot(ids,emotion1,'-', label="Happy", color="green")
    #ax1.plot(ids,emotion2,color='orange', label="Sad")
    plt.xticks(ids, times)
    plt.xlabel('Time')
    plt.ylabel('Percentage (%)')
    plt.legend(bbox_to_anchor=(1,1),loc=1,borderaxespad=0)    

ani = animation.FuncAnimation(fig,animate,interval=1000)
plt.show()   
