from pathlib import Path
from xml.sax.saxutils import escape
import base64
root=Path(__file__).resolve().parent
ns='''xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0" xmlns:xlink="http://www.w3.org/1999/xlink" xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'''
styles=[];items=[]
def box(x,y,w,h,text,size=21,color='#172d40',bold=False):
 key='S'+str(len(styles));styles.append(f'<style:style style:name="{key}" style:family="paragraph"><style:paragraph-properties fo:margin="0cm"/><style:text-properties fo:font-family="Liberation Sans" fo:font-size="{size}pt" fo:color="{color}" fo:font-weight="{"bold" if bold else "normal"}"/></style:style>')
 ps=''.join(f'<text:p text:style-name="{key}">{escape(line)}</text:p>' for line in text.split('\n'))
 items.append(f'<draw:frame draw:style-name="text" svg:x="{x}cm" svg:y="{y}cm" svg:width="{w}cm" svg:height="{h}cm"><draw:text-box>{ps}</draw:text-box></draw:frame>')
def rect(x,y,w,h,style):items.append(f'<draw:rect draw:style-name="{style}" svg:x="{x}cm" svg:y="{y}cm" svg:width="{w}cm" svg:height="{h}cm"/>')
def img(x,y,w,h,path):
 data=base64.b64encode((root/path).read_bytes()).decode();items.append(f'<draw:frame svg:x="{x}cm" svg:y="{y}cm" svg:width="{w}cm" svg:height="{h}cm"><draw:image><office:binary-data>{data}</office:binary-data></draw:image></draw:frame>')
rect(0,0,61,12.55,'navy');rect(2,13.4,57,6.6,'pale')
box(2,1.2,56,1,'NeurReps 2026                                         Symmetry and Geometry in Neural Representations',17,'#ffffff',True)
box(2,3,57,4.7,'Determinants of hyperparameter robustness\nin connectome reservoir computing',54,'#ffffff',True)
box(2,8.6,57,2,'Miles Walter Churchland¹, Raul de Palma Aristides², Jordi Garcia-Ojalvo¹,\nAnna Ritz³, Greg Anderson⁴, Miguel C. Soriano²',22,'#ffffff')
box(2,10.8,57,1,'¹Universitat Pompeu Fabra    ²IFISC, CSIC–UIB    ³Biology, Reed College    ⁴Computer Science, Reed College',17,'#ffffff')
box(2.8,13.85,55,1,'MAIN FINDING',16,'#007d80',True)
box(2.8,14.7,55,2.6,'Higher performance accompanies greater sensitivity\nto hyperparameter choice.',30,'#172d40',True)
box(2.8,17.5,55,2.4,'We compare a C. elegans connectome reservoir with controlled network variants across 96 hyperparameter settings. The connectome occupies a relatively low-variance regime. Changes in excitatory/inhibitory (E/I) edge balance also alter the scaling required by spectral-radius normalization.',21)
box(2,20.8,18,1.4,'01   Connectome as a reservoir',28,'#007d80',True)
img(3.2,22.3,15,8.06,'assets/connectome.png')
box(2,30.7,17.5,15.4,'C. elegans: 297 neurons, 3,604 chemical edges. EleganSign [1].\nOrange: positive    Blue: negative\nLargest connected component shown; arrows omitted.\nFixed recurrent weights; only the readout is trained. Perturb topology, signs, and weights, then normalize spectral radius.\n96 settings × 200 trials\nVary target radius, leak rate, input scale, and bias.\nIPC: information processing capacity; nonlinear recall [2].\nKR: kernel rank; history separation [3].\nMC: memory capacity; recall of the past 30 inputs.\nCV = sd / mean across the 96 settings.\nLower CV = greater robustness.',21)
box(22,20.8,37,1.4,'02   Performance comes with sensitivity',28,'#007d80',True)
img(22,22.5,37,19.51,'assets/architecture_tradeoff.png')
box(22,42.4,37,3.7,'Higher and farther right: more performance, less robustness. Contours enclose the highest-density 50% of trials; markers identify architectures. Color indicates E/I edge balance: 0%, CE ≈24.25%, or 50%.\nCE: C. elegans; ER: Erdős–Rényi; WS: Watts–Strogatz. Sign-pres.: signs preserved; sign shuf.: signs shuffled; pm1: ±1 weights.',21)
rect(20.45,21,.06,24.5,'line');rect(2,46.5,57,.06,'line');rect(32,47.5,.06,28.5,'line')
box(2,47.5,29,1.5,'03   E/I balance traces the tradeoff',28,'#007d80',True)
img(5,49.2,23,24.23,'assets/performance_robustness_curves.png')
box(2,74,29,2.5,'Changing signs moves performance and CV together. Top: IPC, KR, and MC across E/I edge balance; black markers denote empirical balance. Bottom: raw radius and CV averaged across metrics. Points average 200 trials.',21)
box(33,47.5,26,1.5,'04   What the curves show',28,'#007d80',True)
box(33,49.3,26,8.5,'Connectivity and weight magnitudes stay fixed while signs are randomly reassigned at each E/I edge balance.\nNear the all-positive and all-negative endpoints: lower performance and lower CV, indicating greater hyperparameter robustness.\nToward 50% E/I edge balance: higher performance and higher CV, indicating greater sensitivity to hyperparameter choice.',23)
box(33,58,26,1.5,'Why normalization matters',28,'#007d80',True)
box(33,59.7,26,15,'Every perturbed matrix W′ is rescaled:\nWres = (ρtarget / ρraw) W′,    ρraw = max |λᵢ(W′)|.\nThe raw radius decreases toward the balanced regime. Reaching the same target radius therefore requires a larger scaling factor.\nThese curves combine changes in sign balance and placement with normalization-induced rescaling. They do not isolate a causal effect of E/I balance alone.\nReport raw spectral radius alongside architecture comparisons.\nThe full paper finds the same qualitative tradeoff in an edge-matched random topology and after removing unknown or complex-sign edges.',23)
rect(2,76.7,57,3.4,'pale')
box(2.7,77,55,3,'Scope and limitations. Robustness means low relative variation across the tested hyperparameter grid. IPC omits products across different delays. Of 3,604 chemical edges, 1,864 have unknown or complex signs and are assigned signs to match the labeled-edge ratio; selected controls remove them. Electrical synapses and self-loops are excluded; input reaches all units.',21)
box(2,81,57,3,'Selected references    [1] Fenyves et al. (2020), PLOS Comput. Biol. 16, e1007974.\n[2] Dambre et al. (2012), Sci. Rep. 2, 514.\n[3] Roy & Vetterli (2007), EUSIPCO, 606–610.',13,'#4a5e69')
base='''<style:style style:name="text" style:family="graphic"><style:graphic-properties draw:stroke="none" draw:fill="none" fo:padding="0cm"/></style:style><style:style style:name="navy" style:family="graphic"><style:graphic-properties draw:stroke="none" draw:fill="solid" draw:fill-color="#172d40"/></style:style><style:style style:name="pale" style:family="graphic"><style:graphic-properties draw:stroke="none" draw:fill="solid" draw:fill-color="#eaf5f3"/></style:style><style:style style:name="line" style:family="graphic"><style:graphic-properties draw:stroke="none" draw:fill="solid" draw:fill-color="#c7d8db"/></style:style><style:page-layout style:name="layout"><style:page-layout-properties fo:page-width="61cm" fo:page-height="85cm" style:print-orientation="portrait"/></style:page-layout>'''
xml=f'''<?xml version="1.0" encoding="UTF-8"?><office:document {ns} office:version="1.2" office:mimetype="application/vnd.oasis.opendocument.presentation"><office:styles/><office:automatic-styles>{base}{''.join(styles)}</office:automatic-styles><office:master-styles><style:master-page style:name="Default" style:page-layout-name="layout"/></office:master-styles><office:body><office:presentation><draw:page draw:name="Poster" draw:master-page-name="Default">{''.join(items)}</draw:page></office:presentation></office:body></office:document>'''
(root/'poster_editable.fodp').write_text(xml)
