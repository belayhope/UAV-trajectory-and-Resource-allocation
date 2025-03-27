# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# proposed system diffrent parameter 

df = pd.read_csv('reward_data.csv')
# Extract the columns x and y
x = df['Episode']
y = df['Reward-fl']
xx = df['Episode']
yy = df['Reward_nogmm']
xxx = df['Episode']
yyy = df['GMM-MADDPG']
xxxx = df['Episode']
yyyy = df['MADDPG']
xxxxx = df['Episode']
yyyyy = df['DDPG']
xxxxxx = df['Episode']
yyyyyy = df['Greedy']
# Plot the data
plt.figure(figsize=(7, 4))
plt.xlabel('Episodes ')
plt.ylabel('Normalized system reward')
#plt.title('communication coverage')
plt.plot(x, y,  linestyle='-',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xx, yy,  linestyle='-', linewidth=1.5,label='No GMM Fl-MADDPG ')
plt.plot(xxx, yyy, c='g', linestyle='-',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxx, yyyy, c='black', linestyle='-',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyy, c='m', linestyle='-',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxx, yyyyyy, c='chocolate', linestyle='-',linewidth=1.5, label='Greedy ') 
plt.legend()
plt.grid(True)
plt.show()

# coverage
df = pd.read_csv('coverage.csv')
x = df['Episode']
y = df['proposed']
xx = df['Episode']
yy = df['FL-MADDPG']
xxx = df['Episode']
yyy = df['GMM-MADDPG']
xxxx = df['Episode']
yyyy = df['MADDPG']
xxxxx = df['Episode']
yyyyy = df['DDPG']
xxxxxx = df['Episode']
yyyyyy = df['greedy']
# Plot the data
plt.figure(figsize=(6, 4))
plt.xlabel('learning Iteration')
plt.ylabel('Coverage Score')
#plt.title('communication coverage')
plt.plot(x, y,  linestyle='-',linewidth=1.5, label='Proposed Fl-MADDPG')
plt.plot(xx, yy, c='darkgoldenrod' ,linestyle='-', linewidth=1.5,label='No GMM Fl-MADDPG' )
plt.plot(xxx, yyy, c='g', linestyle='-',linewidth=1.5, label='GMM-MADDPG') 
plt.plot(xxxx, yyyy, c='black', linestyle='-',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyy, c='m', linestyle='-',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxx, yyyyyy, c='chocolate', linestyle='-',linewidth=1.5, label='Greedy ') 
plt.ylim([0, 1])  # Set y-axis limits for 0-1 range
plt.legend()
plt.grid(True)
plt.show()
 #AVg datarate
# Extract the columns x and y
df = pd.read_csv('avg_data_rate.csv')
# Extract the columns x and y
x = df['Episode']
y = df['proposed']
xx = df['Episode']
yy = df['FL-MADDPG']
xxx = df['Episode']
yyy = df['GMM-MADDPG']
xxxx = df['Episode']
yyyy = df['MADDPG']
xxxxx = df['Episode']
yyyyy = df['DDPG']
xxxxxx = df['Episode']
yyyyyy = df['greedy']
# Plot the data
plt.figure(figsize=(6, 4))
plt.xlabel('Timeslots ')
plt.ylabel('Average data rate(Mpbs)')
#plt.title('communication coverage')
plt.plot(x, y,  linestyle='-',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xx, yy,  linestyle='-', linewidth=1.5,label='Fl-MADDPG ')
plt.plot(xxx, yyy, c='g', linestyle='-',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxx, yyyy, c='black', linestyle='-',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyy, c='m', linestyle='-',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxx, yyyyyy, c='chocolate', linestyle='-',linewidth=1.5, label='Greedy ') 
plt.legend()
#plt.ylim([2.4, 3.2])
plt.grid(True)
plt.show()

 #fairness
# Extract the columns x and y
df = pd.read_csv('fairness_data.csv')
# Extract the columns x and y
xx = df['Episode']
yy = df['proposed']
xxx = df['Episode']
yyy = df['FL-MADDPG']
xxxx = df['Episode']
yyyy = df['GMM-MADDPG']
xxxxx = df['Episode']
yyyyy = df['MADDPG']
xxxxxx = df['Episode']
yyyyyy = df['DDPG']
xxxxxxx = df['Episode']
yyyyyyy = df['greedy']
#xxxxxx = df['Episode']
#yyyyyy = df['Greedy']
# Plot the data
plt.figure(figsize=(6, 4))
plt.xlabel('learning Iteration ')
plt.ylabel('Firness Index')
#plt.title('communication coverage')
plt.plot(xx, yy,  linestyle='-',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xxx, yyy,  linestyle='-', linewidth=1.5,label='No GMM Fl-MADDPG ')
plt.plot(xxxx, yyyy, c='g', linestyle='-',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxxx, yyyyy, c='black', linestyle='-',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyyy, c='m', linestyle='-',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxxx, yyyyyyy, c='chocolate', linestyle='-',linewidth=1.5, label='Greedy ') 
plt.legend()
plt.ylim([0.2, 1.03])
plt.grid(True)
plt.show()

# Extract the columns x and y
df = pd.read_csv('fairness.csv')
# Extract the columns x and y
xx = df['MIoTD']
yy = df['proposed']
xxx = df['MIoTD']
yyy = df['FL-MADDPG']
xxxx = df['MIoTD']
yyyy = df['GMM-MADDPG']
xxxxx = df['MIoTD']
yyyyy = df['MADDPG']
xxxxxx = df['MIoTD']
yyyyyy = df['DDPG']
xxxxxxx = df['MIoTD']
yyyyyyy = df['greedy']
#xxxxxx = df['Episode']
#yyyyyy = df['Greedy']
# Plot the data
plt.figure(figsize=(5, 4))
plt.xlabel('Number of MIoTDs')
plt.ylabel('Firness Index ')
#plt.title('communication coverage')
plt.plot(xx, yy,  linestyle='-',marker='v',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xxx, yyy,  linestyle='-',marker='v', linewidth=1.5,label='No GMM Fl-MADDPG ')
plt.plot(xxxx, yyyy, c='g', linestyle='-',marker='v',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxxx, yyyyy, c='black', linestyle='-',marker='v',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyyy, c='m', linestyle='-',marker='v',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxxx, yyyyyyy, c='chocolate', linestyle='-',marker='v',linewidth=1.5, label='Greedy ') 
plt.legend()
plt.ylim([0, 1.1])
plt.grid(True)
plt.show()

# Extract the columns x and y
df = pd.read_csv('coveragg.csv')
# Extract the columns x and y
xx = df['MIoTD']
yy = df['proposed']
xxx = df['MIoTD']
yyy = df['FL-MADDPG']
xxxx = df['MIoTD']
yyyy = df['GMM-MADDPG']
xxxxx = df['MIoTD']
yyyyy = df['MADDPG']
xxxxxx = df['MIoTD']
yyyyyy = df['DDPG']
xxxxxxx = df['MIoTD']
yyyyyyy = df['greedy']
#xxxxxx = df['Episode']
#yyyyyy = df['Greedy']
# Plot the data
plt.figure(figsize=(5, 4))
plt.xlabel(' Number of MIoTDs')
plt.ylabel('Coverage Score')
#plt.title('communication coverage')
plt.plot(xx, yy,  linestyle='-',marker='v',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xxx, yyy,  linestyle='-',marker='v', linewidth=1.5,label='No GMM Fl-MADDPG ')
plt.plot(xxxx, yyyy, c='g', linestyle='-',marker='v',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxxx, yyyyy, c='black', linestyle='-',marker='v',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyyy, c='m', linestyle='-',marker='v',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxxx, yyyyyyy, c='chocolate', linestyle='-',marker='v',linewidth=1.5, label='Greedy ') 
plt.legend()
plt.ylim([0, 1.1])
plt.grid(True)
plt.show()

# Extract the columns x and y
df = pd.read_csv('datarate.csv')
# Extract the columns x and y
xx = df['MIoTD']
yy = df['proposed']
xxx = df['MIoTD']
yyy = df['FL-MADDPG']
xxxx = df['MIoTD']
yyyy = df['GMM-MADDPG']
xxxxx = df['MIoTD']
yyyyy = df['MADDPG']
xxxxxx = df['MIoTD']
yyyyyy = df['DDPG']
xxxxxxx = df['MIoTD']
yyyyyyy = df['greedy']
# Plot the data
plt.figure(figsize=(5, 4))
plt.xlabel('Number of MIoTDs ')
plt.ylabel('Average Data Rate(Mpbs)')
#plt.title('communication coverage')
plt.plot(xx, yy,  linestyle='-',marker='v',linewidth=1.5, label='Proposed Fl-MADDPG ')
plt.plot(xxx, yyy,  linestyle='-',marker='v', linewidth=1.5,label='No GMM Fl-MADDPG ')
plt.plot(xxxx, yyyy, c='g', linestyle='-',marker='v',linewidth=1.5, label='GMM-MADDPG ')
plt.plot(xxxxx, yyyyy, c='black', linestyle='-',marker='v',linewidth=1.5, label='MADDPG ') 
plt.plot(xxxxx, yyyyyy, c='m', linestyle='-',marker='v',linewidth=1.5, label='DDPG ') 
#plt.plot(xxxxxxx, yyyyyyy, c='chocolate', linestyle='-',marker='v',linewidth=1.5, label='Greedy ') 
plt.legend()
#plt.ylim([2.4, 3.5])
plt.grid(True)
plt.show()
