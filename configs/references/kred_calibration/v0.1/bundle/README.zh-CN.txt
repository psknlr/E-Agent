KRED 标定参考集 v0.1
审计日期：2026-10-07

适用目的：实验复合物审计、酮还原流程验收、同实验条件内的活性排序，以及活性/选择性区分测试。
交付内容：19个PDB原始mmCIF与验证XML，官方生物装配（以下载manifest成功状态为准），28条动力学、10条选择性、1条近似图读比活，以及2026原始扩展数据。

范围：这是经过来源审计的标定起始参考集 v0.1；包含工业KRED与明确标记的专一底物SDR扩展。不是已验证可泛化的金标准基准。

计数：19个PDB审计条目；28条动力学记录，其中26条有数值、2条ND；10条独立保存的er选择性记录；1条图读比活辅助。记录数不是独立酶或独立实验数。

优先结构：1IPF A用于有条件底物定位，2AE2 A用于有条件产物定位；6ZZO B302 altA为有条件候选。三者都需要保留几何/反应态限制。

严格几何金标：本次审计中没有判定为无需复审的严格精细反应几何gold条目；strict_geometry_gold全部为false。

6ZZO：B302 altA/altB各占有率0.5，RSCC是共同统计；altB与Ser148有2.36Å重原子clash。altA未列clash不等于被独立密度证实。

6ZZP：底物密度较好但Ser148与QT8存在2.11/2.47Å接触；归入review_clash，不直接拟合精细进攻角或距离阈值。

LbADH：1ZK4等五条低配体密度条目保留为审计排除集。1ZK4蛋白1Å分辨率不能覆盖AC0 RSCC0.15/NAP0.22。

反应态：实验NAD+或NADP+、产物态、抑制剂态分别记录。将氧化态辅因子改成还原态后的模型必须另存衍生版本。

生物装配：优先使用提供的官方assembly1并核对其元数据。评分链是位点定位，不代表可删除其他亚基；6XEW口袋含其他亚基Gln247。

配体混合：6XEW HBR/HBS各0.5、互斥；分别准备时共享同一racemate活性记录，不能当作两条独立实测速率。NAD选择A301。

二元模板：4RF2/3/4/5与1Y1P仅作为实验蛋白骨架/辅因子模板；没有目标酮底物实验坐标。Y190F和Ssal M1–M6不获得虚构的突变体晶体证据。

1Y1P辅因子：AMP/NMN/PO4是_struct_conn连接的辅因子片段。保留共价关系、核对氧化还原态，不能因没有单一NDP残基就说辅因子缺失。

序列：保留PDB原始序列、论文报告序列及明确标为derived的序列。Lk标签前缀长20aa；Ssal论文343aa与1Y1P342aa相差首Met。标签与编号不可静默移位。

标签：kcat、Km、kcat/Km、ee/er、固定时间转化和比活是不同量；禁止互相替代或直接拼成一个连续活性值。

缺失：null/空白表示缺失；PNAS的ND是not determined，不是0。PaH150N的Km<5400mM保留为异常界限，禁止用于点值回归。

单位：原始单位与SI换算分列。min^-1除60；mM除1000；uM除1000000；mM^-1min^-1乘1000/60。已发表效率与中心值相除的效率保留为两个字段。

误差：保留论文的±定义。HBDH为fitting error；其余未明确者不重命名为SD或SEM。不能凭±直接做逆方差加权。

Ssal动力学：2024为iPrOH再生体系中的表观动力学，25°C/pH7。M4未达底物饱和，默认只作敏感性分析；M6采用表2±0.8，SI拟合数据支持。

选择性关联：PNAS er1.3属于缺少P194N的九突变变体；与十突变Sph的效率记录不自动合并。

防泄漏：按母本/近缘序列与反应分组。相同酶的链、altloc、不同PDB和近邻突变不能随机拆到训练和标定两侧。

评价：同条件内部做排序/误差；不同标签体系分开报告。小样本可用于流程验收与敏感性测试，不能据此宣称工业KRED泛化能力。

2026扩展：原始109条唯一序列中69条可溶，包含WT；69条1a相对速率，276组2a–5a终点酶底物对。40条不溶记录不能作为无催化活性负例。

2026化学审计：原始SMILES中1a目标位点为醇而图示为酮；5a为羧酸根而图示为乙酯。原值不改写，两个chemical ID暂时隔离。

后处理：本包提供官方原始实验坐标和装配，不声称已完成加氢、质子化、力场准备或几何优化。任何准备结构另存并带来源与变换记录。

许可：PDB为CC0；BRENDA当前为CC BY4.0；2026 Zenodo为CC BY4.0，GitHub保留MIT；其余按sources记录。不要将整包统称CC0。

文件入口
data/structure_manifest.csv — 结构用途、状态与来源
data/activity_labels.csv — 动力学原始值/单位/换算/证据
data/assay_conditions.json — 完整条件与底物/辅酶扫描范围
data/selectivity_labels.csv — er独立记录与Sph变体修正
data/ligand_validation.csv — 逐配体实例密度与局部几何摘要
data/all_reference_sequences.fasta — 原始/报告/衍生参考序列，类别由sequence_manifest解释
data/sources.csv — 原始文献、证据层次与许可说明
evidence/ — 详细来源抽取，含异常/脚注与解释
activity_expansion_2026/ — 原始扩展数据及不改写的化学身份审计

建议起步顺序
1. 先按structure_manifest审查三个有条件候选：1IPF A、2AE2 A、6ZZO B altA。
2. 用HBDH四条WT及四条mutant activity_only做同方向的动力学验收；保留NAD+/NADH差别。
3. 工业KRED排序优先PNAS2015和Ssal2024；对接复合物只作为模型，不作为实验几何真值。
4. 在化学身份和标签类型清理之后再使用2026扩展。

原始出处
HBDH2018_SI | Linear Eyring Plots Conceal a Change in the Rate-Limiting Step in an Enzyme Reaction — Supporting Information | https://doi.org/10.1021/acs.biochem.8b01099.s001
HBDH2018 | Linear Eyring Plots Conceal a Change in the Rate-Limiting Step in an Enzyme Reaction | https://pmc.ncbi.nlm.nih.gov/articles/PMC6300308/
HBDH2020 | Dissecting the Mechanism of (R)-3-Hydroxybutyrate Dehydrogenase by Kinetic Isotope Effects, Protein Crystallography, and Computational Chemistry | https://pmc.ncbi.nlm.nih.gov/articles/PMC7773212/
LbADH2005 | Atomic resolution structures of R-specific alcohol dehydrogenase from Lactobacillus brevis provide the structural bases of its substrate and cosubstrate specificity | https://doi.org/10.1016/j.jmb.2005.04.029
LbADH_BRENDA | BRENDA literature 675348 | https://brenda-enzymes.info/literature.php?e=1.1.1.2&r=675348
LbADH_table_reprint | Analysis of factors influencing enzyme activity and stability in the solid state | https://docserv.uni-duesseldorf.de/servlets/DerivateServlet/Derivate-16498/Doctoral%20thesis_Liliya%20Kulishova.pdf
TRII2003 | Capturing enzyme structure prior to reaction initiation: tropinone reductase-II-substrate complexes | https://doi.org/10.1021/bi0272712
TRII_BRENDA | BRENDA literature 654707 | https://www.brenda-enzymes.org/literature.php?e=1.1.1.236&r=654707
TRII1999 | Structure of tropinone reductase-II complexed with NADP+ and pseudotropine at 1.9 A resolution: implication for stereospecific substrate binding and catalysis | https://doi.org/10.1021/bi9825044
Sm2020 | Phylogenetics-based identification and characterization of a superior 2,3-butanediol dehydrogenase for Zymomonas mobilis expression | https://link.springer.com/article/10.1186/s13068-020-01820-x
PNAS2015 | Origins of stereoselectivity in evolved ketoreductases | https://pmc.ncbi.nlm.nih.gov/articles/PMC4697376/
Ssal2024 | Effective engineering of a ketoreductase for the biocatalytic synthesis of an ipatasertib precursor | https://www.nature.com/articles/s42004-024-01130-5
Ssal2005 | Structure of Sporobolomyces salmonicolor aldehyde reductase with NADPH | https://doi.org/10.1016/j.jmb.2005.07.011
Ortholog2026 | Iterative and data-driven ortholog mining enables reliable discovery of stereoselective ketoreductases | https://www.nature.com/articles/s41467-026-77715-6
Ortholog2026_Zenodo | Public data for Ssal-KRED ortholog mining | https://zenodo.org/records/19574983
Ortholog2026_GitHub | Buller-Lab/Ssal-KRED_orthologs | https://github.com/Buller-Lab/Ssal-KRED_orthologs
wwPDB | PDB archive usage policies | https://www.wwpdb.org/about/usage-policies
TeSADH2022 | Crystallographic snapshots of ternary complexes of thermophilic secondary alcohol dehydrogenase | https://doi.org/10.1002/prot.26339
