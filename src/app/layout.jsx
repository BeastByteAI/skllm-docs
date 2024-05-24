import { Inter } from 'next/font/google'
import localFont from 'next/font/local'
import clsx from 'clsx'

import { Providers } from '@/app/providers'
import { Layout } from '@/components/Layout'

import '@/styles/tailwind.css'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

// Use local version of Lexend so that we can use OpenType features
const lexend = localFont({
  src: '../fonts/lexend.woff2',
  display: 'swap',
  variable: '--font-lexend',
})

export const metadata = {
  title: {
    template: '%s - Docs',
    default: 'Scikit-Learn Meets Large Language Models',
  },
  description:
    'Seamlessly integrate powerful language models into scikit-learn for enhanced NLP tasks.',
}

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      className={clsx('h-full antialiased', inter.variable, lexend.variable)}
      suppressHydrationWarning
    >
      <body className="flex min-h-full bg-white dark:bg-zinc-900">
        <Providers>
          <Layout>{children}</Layout>
        </Providers>
      </body>
    </html>
  )
}
