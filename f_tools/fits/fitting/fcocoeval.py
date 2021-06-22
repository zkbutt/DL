from pycocotools.cocoeval import COCOeval
import numpy as np


class FCOCOeval(COCOeval):

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)

    def summarize(self, catId=None):
        """
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        """

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                ''' 这里是添加的代码 判断是否传入catId，如果传入就计算指定类别的指标 '''
                if isinstance(catId, int):
                    s = s[:, :, catId, aind, mind]
                else:
                    s = s[:, :, :, aind, mind]

            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]

                ''' 这里是添加的代码 判断是否传入catId，如果传入就计算指定类别的指标 '''
                if isinstance(catId, int):
                    s = s[:, catId, aind, mind]
                else:
                    s = s[:, :, aind, mind]

            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])

            ''' 这里是修改 原始只有一个返回值 '''
            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            # return mean_s
            print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            return mean_s, print_string

        stats, print_list = [0] * 12, [""] * 12
        stats[0], print_list[0] = _summarize(1)
        stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

        print_info = "\n".join(print_list)

        if not self.eval:
            raise Exception('Please run accumulate() first')

        return stats, print_info

    def print_clses(self, clses_name):
        # calculate COCO info for all classes
        # coco_stats, print_coco = self.summarize()

        # calculate voc info for every classes(IoU=0.5)
        voc_map_info_list = []
        for i in range(len(clses_name)):
            stats, _ = self.summarize(catId=i)
            voc_map_info_list.append(" {:15}: {:.3f} {:.3f}".format(clses_name[i], stats[1], stats[7]))

        print_voc = "\n".join(voc_map_info_list)
        print(print_voc)

        # # 将验证结果保存至txt文件中
        # with open("record_mAP.txt", "w") as f:
        #     record_lines = ["COCO results:",
        #                     print_coco,
        #                     "",
        #                     "mAP(IoU=0.5) for each category:",
        #                     print_voc]
        #     f.write("\n".join(record_lines))
