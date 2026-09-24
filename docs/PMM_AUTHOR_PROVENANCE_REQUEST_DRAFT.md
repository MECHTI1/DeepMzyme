# PinMyMetal provenance request — ready to send; not sent

To: Heping Zheng <dust.zheng@hotmail.com>
Cc: Wladek Minor <wladek@iwonka.med.virginia.edu>; Nasui Wang <nswang@stu.edu.cn>
Subject: PinMyMetal classmodel row-to-ion mapping and validation folds

Dear Dr. Zheng and colleagues,

We are auditing the released `classmodel_train_set` and `classmodel_test_set`
from PinMyMetal V1.0 (GitHub commit
`59ef46795920322c798db4e5ec500b04451f7904`; Figshare
`10.6084/m9.figshare.25011212.v1`). Could you share a small export keyed by
source side and row number that links each row's original `pdbid`,
`residueid_ion`, and `metalid` to its PDB model, chain, ion residue/atom
identity, and x/y/z coordinates? Residue number, insertion code, and altloc
would help where applicable.

The original ID link matters for `1a0e` train row 4547
(`residueid_ion=887`, `metalid=4637`) and test row 1 (`886`, `4638`), and
`4d8f` train row 5698 (`1316`, `3761`): their deposited structures have
multiple candidate ions, so coordinates alone do not identify the source
rows.

Could you also provide the exact effective metal-classification cohort
(including exclusions, class merging, or resampling) and the original
fivefold validation row assignments, if retained? If the fold IDs are not
retained, the exact ordered rows and split settings would be helpful.

A row-level table is sufficient; we do not need a database dump. Thank you
for your help.
