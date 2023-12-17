#!/usr/bin/env python3.8

import tracing
import copy
import deepening
import models
import torch


def deeper_weight_transfer(teacher, student):
    teacher_blocks = tracing.get_all_deepen_blocks(teacher)
    student_blocks = tracing.get_all_deepen_blocks(student)

    for teacher_info, student_info in zip(teacher_blocks, student_blocks):
        student_hierarchy, student_name = student_info
        teacher_hierarchy, teacher_name = teacher_info
        student_block = getattr(student_hierarchy[-1], student_name)
        teacher_block = getattr(teacher_hierarchy[-1], teacher_name)
        student_block.layers[0] = copy.deepcopy(teacher_block.layers[0])


if __name__ == "__main__":
    teacher = models.ConvNet2()
    deepening.deepen_model(teacher, index=0)
    deepening.deepen_model(teacher, index=0)
    teacher_check = teacher.conv1[0][0].layers[0].weight.mean()

    student = models.ConvNet2()

    deeper_weight_transfer(teacher, student)
    transfer_check = student.conv1.layers[0].weight.mean()

    assert torch.equal(teacher_check, transfer_check)

    teacher = models.ConvNet2()
    deepening.deepen_model(teacher, index=1)
    deepening.deepen_model(teacher, index=1)
    deepening.deepen_model(teacher, index=1)
    teacher_check = teacher.conv2[0][0][0].layers[0].weight.mean()

    student = models.ConvNet2()

    deeper_weight_transfer(teacher, student)

    transfer_check = student.conv2.layers[0].weight.mean()

    assert torch.equal(teacher_check, transfer_check)

    teacher = models.ConvNet2()
    deepening.deepen_model(teacher, index=0)
    deepening.deepen_model(teacher, index=0)
    teacher_check = teacher.conv1[0][0].layers[0].weight.mean()

    student = models.ConvNet2()
    deepening.deepen_model(student, index=0)

    deeper_weight_transfer(teacher, student)
    transfer_check = student.conv1[0].layers[0].weight.mean()

    assert torch.equal(teacher_check, transfer_check)

    print("tests passed")