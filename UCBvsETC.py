import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
def ETC(K,n,m,delta,u_star_list):
    m_min=1
    m_max=int(n/K)
    empirical_means=[0 for i in range(K)]
    num_of_pulls=[0 for i in range(K)]
    optimal_means=u_star_list[:]
    u_opt=max(optimal_means)
    act_regret=0
    emp_regret=0
    #Explore
    for i in range(m):
        for j in range(K):
            reward=np.random.normal(optimal_means[j], 1)
            emp_regret=u_opt-empirical_means[j]
            empirical_means[j]=(reward+num_of_pulls[j]*empirical_means[j])/(1+num_of_pulls[j])
            num_of_pulls[j]+=1
            act_regret+=u_opt-optimal_means[j]
    #Exploit
    exploit_arm_number=empirical_means.index(max(empirical_means))
    for i in range(m*K,n+1):
        reward=np.random.normal(optimal_means[exploit_arm_number], 1)
        emp_regret=u_opt-empirical_means[exploit_arm_number]
        empirical_means[exploit_arm_number]=(reward+num_of_pulls[exploit_arm_number]*empirical_means[exploit_arm_number])/(1+num_of_pulls[exploit_arm_number])
        num_of_pulls[exploit_arm_number]+=1
        act_regret+=u_opt-optimal_means[exploit_arm_number]
    return(act_regret,emp_regret)
def UCB(K,n,u_star_list):
    conf=1
    empirical_means=[0 for i in range(K)]
    ucb_of_means=[float('inf') for i in range(K)]
    num_of_pulls=[0 for i in range(K)]
    optimal_means=u_star_list[:]
    u_opt=max(optimal_means)
    act_regret=0
    emp_regret=0
    for i in range(K):
        exploit_arm_number=i
        reward=np.random.normal(optimal_means[exploit_arm_number], 1)
        empirical_means[i]=reward
        emp_regret=u_opt-empirical_means[exploit_arm_number]
        num_of_pulls[exploit_arm_number]+=1
        act_regret+=u_opt-optimal_means[exploit_arm_number]
        #update ucb of exploited arm
        ucb_of_means[exploit_arm_number]=empirical_means[exploit_arm_number]+math.sqrt((math.log(1/conf)*2)/num_of_pulls[exploit_arm_number])
        # print(ucb_of_means)
    #Exploit
    for i in range(K,n):
        # print("i=",i)
        conf=1/(i**4)
        exploit_arm_number=ucb_of_means.index(max(ucb_of_means))
        reward=np.random.normal(optimal_means[exploit_arm_number], 1)
        emp_regret=u_opt-empirical_means[exploit_arm_number]
        # print("exploit_arm_number=",exploit_arm_number)
        # print("num_of_pulls=",num_of_pulls)
        # print("empirical_means=",empirical_means)
        empirical_means[exploit_arm_number]=(reward+num_of_pulls[exploit_arm_number]*empirical_means[exploit_arm_number])/(1+num_of_pulls[exploit_arm_number])
        num_of_pulls[exploit_arm_number]+=1
        act_regret+=u_opt-optimal_means[exploit_arm_number]
        #update ucb of exploited arm
        ucb_of_means[exploit_arm_number]=empirical_means[exploit_arm_number]+math.sqrt((math.log(1/conf)*2)/num_of_pulls[exploit_arm_number])
        # print(ucb_of_means)
    return(act_regret,emp_regret)

#UCB
K=2
n=1000
m_min=1
m_max=int(n/K)
delta_values = np.arange(0.1, 1.1, 0.1)
actual_regret=np.zeros_like(np.arange(0.1, 1.1, 0.1))
emp_regret=np.zeros_like(np.arange(0.1, 1.1, 0.1))
theoretical_bounds = []
for delta in delta_values:
    term1=n*delta
    term2=math.log((n*delta*delta)/4)
    term2=max(0,term2)
    term2+=1
    term2=(4*term2)/delta
    term2+=delta
    theoretical_bounds.append(min(term1,term2))
for ep in tqdm(range(100)):
    ep_actual_regret=[]
    ep_emp_regret=[]
    delta_list=[]
    for delta in delta_values:
        m_opt=max(1, int(4 / delta**2 * np.log(n * delta**2 / 4)))
        u_star_list=[0,-1*delta]
        cummulative_act_regret,cummulative_emp_regret=UCB(K,n,u_star_list)
        ep_actual_regret.append(cummulative_act_regret)
        ep_emp_regret.append(cummulative_emp_regret)
        delta_list.append(delta)
    actual_regret=(ep*actual_regret+ep_actual_regret)/(ep+1)
    emp_regret=(ep*emp_regret+ep_emp_regret)/(ep+1) 
plt.figure(figsize=(10, 6))
UCB_regret=actual_regret[:]
#plt.plot(delta_values, actual_regret, label="UCB", color="Black")
#ETC
del(actual_regret)
regret_for_m=[]
m_values=[25,50,75,100,125,150,175,200,225,250]
for m_x in range(len(m_values)):
    K=2
    n=1000
    m_min=1
    m_max=int(n/K)
    delta_values = np.arange(0.1, 1.1, 0.1)
    actual_regret=np.zeros_like(np.arange(0.1, 1.1, 0.1))
    emp_regret=np.zeros_like(np.arange(0.1, 1.1, 0.1))
    theoretical_bounds = []
    for delta in delta_values:
        term1=n*delta
        term2=math.log((n*delta*delta)/4)
        term2=max(0,term2)
        term2+=1
        term2=(4*term2)/delta
        term2+=delta
        theoretical_bounds.append(min(term1,term2))
    for ep in tqdm(range(100)):
        ep_actual_regret=[]
        ep_emp_regret=[]
        delta_list=[]
        for delta in delta_values:
            m_opt= m_values[m_x]#max(1, int(4 / delta**2 * np.log(n * delta**2 / 4)))
            u_star_list=[0,-1*delta]
            cummulative_act_regret,cummulative_emp_regret=ETC(K,n,m_opt,delta,u_star_list)
            ep_actual_regret.append(cummulative_act_regret)
            ep_emp_regret.append(cummulative_emp_regret)
            delta_list.append(delta)
        actual_regret=(ep*actual_regret+ep_actual_regret)/(ep+1)
        emp_regret=(ep*emp_regret+ep_emp_regret)/(ep+1)   
    regret_for_m.append(actual_regret)             
#plt.figure(figsize=(10, 6))
# line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
# for i in range(len(regret_for_m)):
#     plt.plot(delta_values, regret_for_m[i], label="ETC with m="+str(m_values[i]), color=line_colors[i])
# plt.xlabel("Delta")
# plt.ylabel("Expected Regret")
# plt.legend()
# plt.title("Expected Regret vs. Delta of UCB/ETC Algorithm")
# #plt.grid(True)
# plt.show()

line_colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']
for i in range(len(regret_for_m)):
    plt.plot(delta_values, np.subtract(UCB_regret,regret_for_m[i]), label="E[UCB - ETC with m="+str(m_values[i])+"]", color=line_colors[i])
plt.xlabel("Delta")
plt.ylabel("Expected Regret(UCB-ETC)")
plt.legend()
plt.title("Expected Regret of UCB(c=1/t^4) - ETC (with varying m) vs. Delta")
#plt.grid(True)
plt.show()


    