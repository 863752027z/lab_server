import datetime


log_file = open('/home/zlw/PycharmProjects/pycharm.log', 'a+')
time_now = str(datetime.datetime.now()) + '\n'
content = 'out1, out1.1, out1.2 num_epoch = 100, 50, 40. The direct is to save moudle.'
log_file.write(time_now + content + '\n')