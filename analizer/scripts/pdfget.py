
import PyPDF2 as pypdf
import requests
from io import BytesIO, StringIO


def enc_fixer(t, to='latin-1', frm='cp1251'):
    return t.encode(to, 'ignore').decode(frm)


# please see
# http://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words
def find_words(instring, prefix='', words=None):
    if not instring:
        return []
    if words is None:
        words = set()
        with open('/usr/share/dict/words') as f:
            for line in f:
                words.add(line.strip())
    if (not prefix) and (instring in words):
        return [instring]
    prefix, suffix = prefix + instring[0], instring[1:]
    solutions = []
    # Case 1: prefix in solution
    if prefix in words:
        try:
            solutions.append([prefix] + find_words(suffix, '', words))
        except ValueError:
            pass
    # Case 2: prefix not in solution
    try:
        solutions.append(find_words(suffix, prefix, words))
    except ValueError:
        pass
    if solutions:
        return sorted(solutions,
                      key=lambda solution: [len(word) for word in solution],
                      reverse=True)[0]
    else:
        raise ValueError('no solution')


if __name__ == '__main__':
    pdf_urls = ['http://www.uic.unn.ru/~zny/ml/Lectures/ml_pres.pdf',
    'http://www.botik.ru/~psi/PSI/disk_20/e-book/e-book/2-5/'
            '03-Kormalev-Prilozhenija-metodov-p-35.pdf',
    'https://compscicenter.ru/media/slides/machine_learning_1_2014_autumn'
            '/2014_09_12_machine_learning_1_2014_autumn.pdf',
    'http://www.iitp.ru/upload/publications/6256/vyugin1.pdf',
    'http://courses.graphicon.ru/files/courses/vision/2009/cv_2009_06.pdf',
    'http://www.sciteclibrary.ru/texsts/rus/stat/st4820.pdf',
    'http://sis.khashaev.ru/data/2013/july/SIS_machinelearning_july.pdf',
    'http://rcdl2013.uniyar.ac.ru/doc/full_text/pd_1.pdf']
    pdf_texts = []
    for url in pdf_urls:
        r = requests.get(url)
        BytesIO(r.content)
        bytes = BytesIO(r.content)
        pdfreader = pypdf.PdfFileReader(bytes)
        pages = []
        for page in pdfreader.pages:
            pages.append(page.extractText())
        text = page.extractText()
        pages = list(map(enc_fixer, pages))

        # concatenate all text to one object
        buf = StringIO()
        for p in pages:
            buf.write(p)
        pdf_texts.append(buf.getvalue())
        t = buf.getvalue()
