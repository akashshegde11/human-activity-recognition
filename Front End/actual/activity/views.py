from django.shortcuts import render , HttpResponse
import os


def start(request):
    return render(request, 'activity/page1.html')

def page2(request):
    return render(request, 'activity/page2.html')

def page3(request):
    return render(request, 'activity/page3.html')

def takeinput(request):
    if request.method == 'POST':
        training1 = request.POST.get('training1')
        training2 = request.POST.get('training2')
        test1 = request.POST.get('test1')
        test2 = request.POST.get('test2')
        epoch = request.POST.get('epoch')
        output = []
        output = script(training1, training2, test1, test2, epoch)
        return render(request, 'activity/exe.html', {
            'training1': training1,
            'training2': training2,
            'test1': test1,
            'test2': test2,
            'epoch': epoch,
            'output': output,
        })
    return render(request, 'activity/page4.html', {})


def script(training1,training2,test1,test2,epoch):
    f=open('mit.txt','w+')
    f.write(training1)
    f.write('\n')
    f.write(training2)
    f.write('\n')
    f.write(test1)
    f.write('\n')
    f.write(test2)
    f.write('\n')
    f.write(epoch)
    f.write('\n')
    f.close()
    os.system('python D:\\Documents\\actual\\activity\\lstm.py')
    return training1,training2,test1,test2,epoch


