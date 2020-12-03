def cre_pdatas(file_txt, p_boxes, p_labels_name, p_scores):
    '''
    face 0.8658207058906555 542 355 581 400
    :param path_save:
    :param file_name:
    :param p_boxes:  ltrb
    :param p_labels_name: 类别名称
    :param p_scores:
    :return:
    '''
    # tolist

    lines_write = []
    for label_name, score, bbox in zip(p_labels_name, p_scores, p_boxes):
        _bbox = [str(i.item()) for i in bbox.tolist()]
        bbox_str = ' '.join(_bbox)
        _line = label_name + ' ' + score + ' ' + bbox_str + '\n'
        lines_write.append(_line)

    with open(file_txt, "w") as f:
        f.writelines(lines_write)


if __name__ == '__main__':
    pass
