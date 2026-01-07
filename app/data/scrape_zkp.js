import puppeteer from "puppeteer";
import fs from 'node:fs'

const subArticleMap = {
    'a': 1,
    'b': 2,
    'c': 3,
    'č': 4,
    'd': 5,
    'e': 6,
    'f': 7,
    'g': 8,
    'h': 9
}

function writeArticles(articles){
    fs.writeFile('./data/articles.json', JSON.stringify(articles), err => {
        if(err){
            console.error(err)
        }else{
            console.log('File written successfully!')
        }
    })
}

writeArticles({asdf: 'asdf'})

async function scrape(){
    const browser = await puppeteer.launch({headers: false})
    const page = await browser.newPage()
    await page.goto('https://www.uradni-list.si/glasilo-uradni-list-rs/vsebina/2012-01-1405')

    const data = await page.evaluate((subArticleMap) => {
        const container = document.querySelector('.col-sm-12 .content-segments')
        const aSegments = Array.from(container.querySelectorAll('.esegment_a'))

        const articles = []

        aSegments.forEach(aSegment => {
            const nextNode = aSegment.nextElementSibling
            const aSegmentText = aSegment.textContent.replaceAll('\t', '').replaceAll('\n', '').trim()

            if(nextNode && nextNode.classList.contains('esegment_p') && aSegmentText.endsWith('člen')){
                let articleNumber =  aSegmentText.replace(' člen', '')
                if(articleNumber.endsWith('.')){
                    articleNumber = articleNumber.slice(0, -1)
                }

                const [int, letter] = articleNumber.split('.')

                let articleIndex = parseInt(int)

                if(letter && Object.keys(subArticleMap).includes(letter)){
                    articleIndex += subArticleMap[letter] / 10
                }

                let text = ''

                if(nextNode.classList.contains('esegment_p')){
                    const clone = nextNode.cloneNode(true)

                    clone.querySelectorAll('br').forEach(br => {
                        br.replaceWith('\n')
                    })

                    text = clone.textContent.replaceAll('\t\n', '').replaceAll('\t', '')
                }

                _id = `zkp_${articleIndex}` + (letter ? '' : '.0')

                if(articles.some(a => a._id === _id)) return 

                articles.push(
                    {
                        _id,
                        law_id: 'zkp',
                        article_number: articleNumber,
                        article_index: articleIndex,
                        text,
                        language: 'sl'
                    }
                )
            }
        })
        return articles
        
    }, subArticleMap)

    writeArticles(data)
    await browser.close()
}

scrape()
