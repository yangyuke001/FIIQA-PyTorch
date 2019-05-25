import torch 
from torch.utils.data import DataLoader
from torch.autograd import Variable

from shufflenetv2 import ShuffleNetV2
import torchvision
import torchvision.transforms as transforms
from datagen import ListDataset
import csv
import os
from PIL import Image

test_data = './data/test-faces/'
list_file = './data/new_4people_test_standard.txt'
checkpoint = torch.load('./checkpoint/97_160_2.pth')

def write_csv(header, write_data, filename):
    # header-标题 write_data-写入数据 filename-文件名 
    with open(filename, 'a', newline='',encoding='utf-8-sig') as csvFile:
        writer = csv.writer(csvFile)
        if os.path.getsize(filename) == False:
            # 先写columns_name
            writer.writerow(header)
        # 写入多行用writerows
        writer.writerows(write_data)

def test():
    # configure model
    net= ShuffleNetV2(input_size=160)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    transform_test = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    testset = ListDataset(root=test_data, list_file=list_file, \
        transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, \
        shuffle=False, num_workers=4, pin_memory = True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    results = []
    for ii,(data,path) in enumerate(testloader):

        #input = torch.autograd.Variable(data)
        with torch.no_grad():
            score = net(data)
        
        probability = torch.nn.functional.softmax(score,dim=1)#[:,0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()
        expect = torch.sum(torch.arange(0,200).float()*probability, 1)
        #print('expect: %.4f' % expect.numpy())
        batch_results = [(path_,expect_) for path_,expect_ in zip(path,expect) ]
        print('batch_results: ',batch_results)
        results += batch_results
    write_csv('FIIQA',results,'result.csv')

    return results


if __name__ == "__main__":
    """Testing
    """
    test()