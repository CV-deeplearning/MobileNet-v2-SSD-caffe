#coding:utf-8

import matplotlib.pyplot as plt

def log_info(log_file):
    loss_list = []
    acc_list = []
    with open(log_file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            if len(words) != 13:
                continue
            if words[7][-1] != ',':
                continue
            loss = float(words[7].split(',')[0])
            acc = float(words[9].split(',')[0])
            loss_list.append(loss)
            acc_list.append(acc)

    return loss_list, acc_list


def draw_info(loss_list, yname='loss', save_name='loss.jpg'):
    x = list(range(len(loss_list)))
    plt.xlabel("iter")#x轴上的名字
    plt.ylabel(yname)#y轴上的名字
    plt.plot(x, loss_list,  color='r',markerfacecolor='blue',marker='o')
    plt.legend()
    #plt.show()
    plt.savefig(save_name)
    plt.close()


loss_list, acc_list = log_info('log.txt')
draw_info(loss_list)
draw_info(acc_list, 'acc', 'acc.jpg')



