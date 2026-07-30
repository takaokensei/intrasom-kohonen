# Citation Audit & Review Checklist (`paper/references.bib`)

This checklist documents the 100% verification pass performed on all 17 citations in `paper/references.bib` to ensure academic integrity and precise alignment between text claims and bibliographic sources.

| Key | Authors & Year | Claim Supported in `paper/main.tex` | Verification Source / Link | Status |
|---|---|---|---|---|
| `kohonen1982self` | Kohonen (1982) | Original SOM formulation (Biological Cybernetics) | DOI: [10.1007/BF00337288](https://doi.org/10.1007/BF00337288) | Verified |
| `kohonen2001self` | Kohonen (2001) | SOM monograph & topology preservation | Springer Series in Info. Sci. (ISBN 978-3-540-67921-9) | Verified |
| `kohonen2013essentials` | Kohonen (2013) | Batch SOM training & neighborhood updating rule | DOI: [10.1016/j.neunet.2012.09.018](https://doi.org/10.1016/j.neunet.2012.09.018) | Verified |
| `ultsch1993self` | Ultsch (1993) | Original U-matrix concept for SOM visualization | DOI: [10.1007/978-3-642-50974-2_31](https://doi.org/10.1007/978-3-642-50974-2_31) | Verified |
| `costa2007segmentacao` | Costa & Netto (2007) | Inter-neuron U-matrix math & $1/\sqrt{2}$ diagonal factor (Eq. 3) | DOI: [10.1590/S0103-17592007000200002](https://doi.org/10.1590/S0103-17592007000200002) (SciELO) | Verified |
| `vesanto2000clustering` | Vesanto & Alhoniemi (2000) | Reduced U-matrix calculation per neuron & SOM clustering | DOI: [10.1109/72.846731](https://doi.org/10.1109/72.846731) | Verified |
| `alcock1999time` | Alcock & Manolopoulos (1999) | Source of Synthetic Control Chart Time Series dataset | PCI 1999 proceedings, Ioannina, Greece (UCI Repository origin) | Verified |
| `kaski1996comparing` | Kaski & Lagus (1996) | SOM quality evaluation & comparison methods | DOI: [10.1007/3-540-61510-5_138](https://doi.org/10.1007/3-540-61510-5_138) | Verified |
| `flexer2001use` | Flexer (2001) | Variations in U-matrix implementation across SOM tools | DOI: [10.3233/IDA-2001-5504](https://doi.org/10.3233/IDA-2001-5504) | Verified |
| `sculley2015hidden` | Sculley et al. (2015) | Technical debt and hidden assumptions in ML systems | NeurIPS 2015, pp. 2503-2511 | Verified |
| `stodden2014implementing` | Stodden & Miguez (2014) | Best practices in computational science & software infrastructure | DOI: [10.5334/jors.ay](https://doi.org/10.5334/jors.ay) | Verified |
| `peng2011reproducible` | Peng (2011) | Reproducible research in computational science | DOI: [10.1126/science.1213847](https://doi.org/10.1126/science.1213847) | Verified |
| `sandve2013ten` | Sandve et al. (2013) | Ten simple rules for reproducible computational research | DOI: [10.1371/journal.pcbi.1003285](https://doi.org/10.1371/journal.pcbi.1003285) | Verified |
| `ultsch1990innc` | Ultsch & Siemon (1990) | Early U-matrix presentation at INNC'90 | INNC'90 proceedings, Kluwer, Vol. 1, pp. 305-308 | Verified |
| `gouvea2023intrasom` | Gouvêa et al. (2023) | Official publication for IntraSOM library (InTRA-USP) | DOI: [10.1016/j.simpa.2023.100570](https://doi.org/10.1016/j.simpa.2023.100570) (*Software Impacts*) | Verified |
| `wehrens2007self` | Wehrens & Buydens (2007) | R `kohonen` package U-matrix implementation reference | DOI: [10.18637/jss.v021.i05](https://doi.org/10.18637/jss.v021.i05) | Verified |
| `venna2007visualizing` | Venna & Kaski (2007) | Topology preservation & visualization in SOMs | DOI: [10.1016/j.neunet.2006.11.013](https://doi.org/10.1016/j.neunet.2006.11.013) | Verified |

---

## Explicit Rules Enforced
1. `alcock1999time` is cited as the dataset source for *Synthetic Control*, replacing `kaski1996comparing`.
2. `sculley2015hidden` and `sandve2013ten` are cited strictly for general technical debt and ML reproducibility, NOT as sources for `ENGINEERING_AUDIT.md`.
3. `ultsch1990innc` is renamed from `ultsch2003maps` to match its actual publication year (INNC'90).
4. `gouvea2023intrasom` replaces `intrasom2026pypi`, citing the peer-reviewed *Software Impacts* paper.
