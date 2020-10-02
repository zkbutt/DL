import re

import numpy as np

import logging
from logging.handlers import RotatingFileHandler  # 按文件大小滚动备份
import colorlog  # 控制台日志输入颜色
import time
import datetime
import os

import sys
import os

import os
import sys
import logging

import matplotlib.pyplot as plt

s = '''
<li class="treeview active">
    <a href="#">
        <i class="fa fa-yen"></i>
        <span>应收账查询</span>
        <i class="fa fa-angle-left text-bold pull-right "></i>
    </a>

    <!--浜岀骇鑿滃崟  -->
    <ul class="treeview-menu menu-open" style="display: block;">
        <li>
            <a href="/crm/ui/paymentSearch">
                <i class="fa fa-circle-o"></i>
                <span>合同应收账查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/detailSearch">
                <i class="fa fa-circle-o"></i>
                <span>合同明细账查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payHostDetails">
                <i class="fa fa-circle-o"></i>
                <span>主机回款明细</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/balanceSearch">
                <i class="fa fa-circle-o"></i>
                <span>应收账余额查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payPartDetails">
                <i class="fa fa-circle-o"></i>
                <span>备件回款明细</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payIncomeAnalysis">
                <i class="fa fa-circle-o"></i>
                <span>回款分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/analyzeStaff">
                <i class="fa fa-circle-o"></i>
                <span>人员回款类型分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/analyzeArea">
                <i class="fa fa-circle-o"></i>
                <span>片区回款类型分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidMainframeInternal">
                <i class="fa fa-circle-o"></i>
                <span>内勤-主机到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payBackToArticle">
                <i class="fa fa-circle-o"></i>
                <span>市场部回款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidMainframe">
                <i class="fa fa-circle-o"></i>
                <span>片区-主机到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidPartsInternal">
                <i class="fa fa-circle-o"></i>
                <span>内勤-备件到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidMainframeStaff">
                <i class="fa fa-circle-o"></i>
                <span>人员-主机到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidPartsRepairs">
                <i class="fa fa-circle-o"></i>
                <span>片区-备件维修到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paidParts">
                <i class="fa fa-circle-o"></i>
                <span>人员-备件到款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payBillingHistory">
                <i class="fa fa-circle-o"></i>
                <span>合同开票历史</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payIncome">
                <i class="fa fa-circle-o"></i>
                <span>销售收入统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payAgingAccount">
                <i class="fa fa-circle-o"></i>
                <span>账龄台帐</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payRegulationAgingAccount">
                <i class="fa fa-circle-o"></i>
                <span>新增减账龄台帐</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paymentNode">
                <i class="fa fa-circle-o"></i>
                <span>应收账节点查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/ViFAMDiffSum">
                <i class="fa fa-circle-o"></i>
                <span>应收账款差异查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/yearTotalOrderOfnReceivedPrice">
                <i class="fa fa-circle-o"></i>
                <span>成套处-按产品分类全年总回款</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/yearTotalOrder">
                <i class="fa fa-circle-o"></i>
                <span>成套处-按产品分类全年订货</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/viContractWithProductKind">
                <i class="fa fa-circle-o"></i>
                <span>合同按产品类型分类查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/viContractWithCheckKind">
                <i class="fa fa-circle-o"></i>
                <span>合同按考核类型分类查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/monthOrderAssessment">
                <i class="fa fa-circle-o"></i>
                <span>成套处-项目订货月考核</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/paymentAndInvoiceYearInfo">
                <i class="fa fa-circle-o"></i>
                <span>全年起票回款统计</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/accountsReceivableByMonth">
                <i class="fa fa-circle-o"></i>
                <span>应收账及回款明细查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/accountsReceivableCollection">
                <i class="fa fa-circle-o"></i>
                <span>应收款汇总分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/areaOrderView">
                <i class="fa fa-circle-o"></i>
                <span>销售处-片区任务完成情况</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/personOrderView">
                <i class="fa fa-circle-o"></i>
                <span>销售处-片员任务完成情况</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/areaPersonOrderAndPaymentView">
                <i class="fa fa-circle-o"></i>
                <span>销售处-片区人员完成情况</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/report/areaBJOrderView">
                <i class="fa fa-circle-o"></i>
                <span>销售处-备件订货月报表</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/monthOrderPayment">
                <i class="fa fa-circle-o"></i>
                <span>成套处-回款月考核表</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/monthOrderAssessmentBJ">
                <i class="fa fa-circle-o"></i>
                <span>成套处-备件订货月考核</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/paymentBackByMonth">
                <i class="fa fa-circle-o"></i>
                <span>货款分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/payHistory">
                <i class="fa fa-circle-o"></i>
                <span>合同回款历史</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/CARBTArea">
                <i class="fa fa-circle-o"></i>
                <span>应收款按时间点汇总分析</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/accountsReceivableByTime">
                <i class="fa fa-circle-o"></i>
                <span>应收账款按时间点明细查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/agingledgerforfinance">
                <i class="fa fa-circle-o"></i>
                <span>财务处-账龄台账查询</span>
            </a>
        </li>
        <li>
            <a href="/crm/ui/accountsReceivableWithYear">
                <i class="fa fa-circle-o"></i>
                <span>应收账款按年综合分析</span>
            </a>
        </li>
    </ul>
    <!--  涓夌骇鑿滃崟-->

</li>
'''

pattern = re.compile(r' <span>(.*)</span>')
ret = pattern.findall(s, )

# search = re.search(' <span>.*</span>', s, re.M)
print(ret)
