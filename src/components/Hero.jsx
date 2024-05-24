import { Fragment } from 'react'
import Image from 'next/image'
import clsx from 'clsx'
import { Highlight, themes } from 'prism-react-renderer'


import { Button } from '@/components/Button'
import { HeroBackground } from '@/components/HeroBackground'
import blurOrangeImage from '@/images/blur-orange-3.png'
import blurOrangeImage2 from '@/images/blur-orange-light.png'
import blurRedImage from '@/images/blur-red-2.png'

const codeLanguage = 'python'
const code = 
`from skllm.models.gpt.classification.zero_shot import 
ZeroShotGPTClassifier

clf = ZeroShotGPTClassifier(model="gpt-4-turbo")
clf.fit(X,y)
labels = clf.predict(X)`

const tabs = [
  { name: 'app.py', isActive: true },
  // { name: 'build.py', isActive: false },
]

function TrafficLightsIcon(props) {
  return (
    <svg aria-hidden="true" viewBox="0 0 42 10" fill="none" {...props}>
      <circle cx="5" cy="5" r="4.5" />
      <circle cx="21" cy="5" r="4.5" />
      <circle cx="37" cy="5" r="4.5" />
    </svg>
  )
}

export function Hero() {
  // Function to handle scroll up by 10 pixels
  const handleScrollToElement = (event) => {
    event.preventDefault(); // Prevent default anchor link behavior
    const targetId = event.currentTarget.getAttribute('href').substring(1); // Extract the target id from href
    const targetElement = document.getElementById(targetId);

    if (targetElement) {
      const targetPosition = targetElement.getBoundingClientRect().top + window.scrollY - 20;
      window.scrollTo({ top: targetPosition, behavior: 'smooth' });
    }
  };
  return (
    // COMMENT: THE WHOLE HEADER BACKGROUND 
    <div className="overflow-hidden bg-zinc-900 dark:-mb-32 dark:mt-[-4.75rem] dark:pb-32 dark:pt-[4.75rem]"> 
      <div className="py-16 sm:px-2 lg:relative lg:px-0 lg:py-20">
        <div className="mx-auto grid max-w-2xl grid-cols-1 items-center gap-x-8 gap-y-16 px-4 lg:max-w-8xl lg:grid-cols-2 lg:px-8 xl:gap-x-16 xl:px-12">
          <div className="relative z-10 md:text-center lg:text-left">
            <Image
              className="absolute bottom-full right-full -mb-56 -mr-72 opacity-30"
              src={blurOrangeImage}
              alt=""
              width={530}
              height={530}
              unoptimized
              priority
            />
            <div className="relative">
              {/* GRADIENT HEADER SLOGAN */}
              <p className="inline bg-gradient-to-r from-orange-300 via-orange-500 to-orange-300 bg-clip-text font-display text-5xl tracking-tight text-transparent">
              Scikit-Learn Meets Large Language Models
              </p>
              <p className="mt-3 text-2xl tracking-tight text-zinc-400">
              Seamlessly integrate powerful language models into scikit-learn for enhanced NLP tasks.
              </p>
              <div className="mt-8 flex gap-4 md:justify-center lg:justify-start">
                <Button href="#cde" onClick={handleScrollToElement}>Get started</Button>
                <Button href="https://github.com/iryna-kondr/scikit-llm" variant="secondary">
                  View on GitHub
                </Button>
              </div>
            </div>
          </div>
          <div className="relative lg:static xl:pl-10">
            <div className="absolute inset-x-[-50vw] -bottom-48 -top-32 [mask-image:linear-gradient(transparent,white,white)] lg:-bottom-32 lg:-top-32 lg:left-[calc(50%+14rem)] lg:right-0 lg:[mask-image:none] dark:[mask-image:linear-gradient(transparent,white,transparent)] lg:dark:[mask-image:linear-gradient(white,white,transparent)]">
              <HeroBackground className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 lg:left-0 lg:translate-x-0 lg:translate-y-[-60%]" />
            </div>
            <div className="relative">
              <Image
                className="absolute -right-64 -top-64 opacity-30"
                src={blurOrangeImage}
                alt=""
                width={530}
                height={530}
                unoptimized
                priority
              />
              <Image
                className="absolute -bottom-40 -right-44 opacity-30"
                src={blurOrangeImage2}
                alt=""
                width={567}
                height={567}
                unoptimized
                priority
              />
              {/* COMMENT: CODE BLUR ACROSS A RECTANGLE LEFT PART */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-orange-500 via-orange-500/70 to-orange-500 opacity-10 blur-lg" />
              {/* COMMENT: CODE BLUR ACROSS A RECTANGLE RIGHT PART */}
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-tr from-orange-500 via-orange-500/70 to-orange-500 opacity-10" />
              {/* COMMENT: CODE BLOCK BACKGROUND BLURRED */}
              <div className="relative rounded-2xl bg-zinc-900/80 ring-1 ring-white/10 backdrop-blur">
                {/* COMMENT: CODE BLUR ACROSS A RECTANGLE UPPER PART */}
                <div className="absolute -top-px left-20 right-11 h-px bg-gradient-to-r from-orange-300/0 via-orange-300/70 to-orange-300/0" />
                {/* COMMENT: CODE BLUR ACROSS A RECTANGLE LOWER PART */}
                <div className="absolute -bottom-px left-11 right-20 h-px bg-gradient-to-r from-orange-300/0 via-orange-300/70 to-orange-300/0" />
                <div className="pl-4 pt-4">
                  {/* COMMENT: THREE DOTS CODE BLOCK */}
                  <TrafficLightsIcon className="h-2.5 w-auto stroke-zinc-400/30" />
                  <div className="mt-4 flex space-x-2 text-xs">
                    {tabs.map((tab) => (
                      <div
                        key={tab.name}
                        className={clsx(
                          'flex h-6 rounded-full',
                          tab.isActive
                            // COMMENT: CODE BLOCK APP.PY
                            ? 'bg-gradient-to-r from-orange-300/30 via-orange-300 to-orange-300/30 p-px font-medium text-orange-300'
                            // COMMENT: CODE BLOCK BUILD.PY
                            : 'text-zinc-400',
                        )}
                      >
                        <div
                          className={clsx(
                            'flex items-center rounded-full px-2.5',
                            // COMMENT: BUTTON APP.PY BACKGROUND
                            tab.isActive && 'bg-zinc-800',
                          )}
                        >
                          {tab.name}
                        </div>
                      </div>
                    ))}
                  </div>
                  <div className="mt-6 flex items-start px-1 text-sm">
                    <div
                      aria-hidden="true"
                      // COMMENT: CODE BLOCK INTEGERS
                      className="select-none border-r border-zinc-400/5 pr-4 font-mono text-zinc-600"
                    >
                      {Array.from({
                        length: code.split('\n').length,
                      }).map((_, index) => (
                        <Fragment key={index}>
                          {(index + 1).toString().padStart(2, '0')}
                          <br />
                        </Fragment>
                      ))}
                    </div>
                    <Highlight

                      code={code}
                      language={codeLanguage}
                      theme={{ plain: {color:'#e4e4e7', fontStyle:'normal', }, 
                      styles: [
                        // COMMENT: FROM, IMPORT COLORS
                        {
                          types: ['builtin', 'changed', 'keyword'],
                          style: {
                            color: '#ff7500' // orange
                          }
                        },
                        // COMMENT: GPT-3.5 COLORS
                        {
                          types: ['string'],
                          style: {
                            color: '#fdba74' // orange-300
                          }
                        },
                        {
                          types: ['property', 'punctuation', 'operator'],
                          style: {
                            color: '#a1a1aa' // zinc-400
                          }
                        },
                      ]
                    }}
                    >
                      {({
                        className,
                        style,
                        tokens,
                        getLineProps,
                        getTokenProps,
                      }) => (
                        <pre
                          className={clsx(
                            className,
                            'flex overflow-x-auto pb-6',
                          )}
                          style={style}
                        >
                          <code className="px-4">
                            {tokens.map((line, lineIndex) => (
                              <div key={lineIndex} {...getLineProps({ line })}>
                                {line.map((token, tokenIndex) => (
                                  <span
                                    key={tokenIndex}
                                    {...getTokenProps({ token })}
                                  />
                                ))}
                              </div>
                            ))}
                          </code>
                        </pre>
                      )}
                    </Highlight>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
