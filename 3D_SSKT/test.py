import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import sys
import json

from utils import AverageMeter


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []

    for i in range(sorted_scores.size(0)):
    
        video_results.append({
            'label': class_names[int(locs[i])],
            'score': float(sorted_scores[i])
        })

    test_results['results'][video_id] = video_results


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}, 'Accuracy' : {}}
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            inputs = Variable(inputs)
            outputs, auxiliary_image_output, auxiliary_places_output = model(inputs)
          
            outputs = F.softmax(outputs)

            for j in range(outputs.size(0)):
                if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                    calculate_video_results(output_buffer, previous_video_id,
                                            test_results, class_names)
                    output_buffer = []
                output_buffer.append(outputs[j].data.cpu())
                previous_video_id = targets[j]

            if (i % 100) == 0:
                with open(
                        os.path.join(opt.result_path, '{}.json'.format(
                            'test')), 'w') as f:
                    json.dump(test_results, f)

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('[{}/{}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                    i + 1,
                    len(data_loader),
                    batch_time=batch_time,
                    data_time=data_time))
        
    total = 0.0
    count = 0.0
    for video_name in test_results['results'].keys():
        if test_results['results'][video_name][0]['label'].lower() in video_name.lower():
            count += 1
        total += 1

    print('accuracy: ', count / total)
    test_results['Accuracy'] = count / total
    with open(os.path.join(opt.result_path, '{}.json'.format('test')),'w') as f:
        json.dump(test_results, f)
