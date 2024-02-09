import PageLayout from '@/components/layout/page-layout'
import PaymentMethodsList from '@/components/organization/payment-methods-list'
import { ContentBox } from '@/components/ui/content-box'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { useAppContext } from '@/contexts/app-context'
import useUsage from '@/hooks/api/billing/useUsage'
import { isEnterprise } from '@/lib/domain/billing'
import { toDateCycle, toDollars } from '@/lib/utils'
import { Info } from 'lucide-react'
import Head from 'next/head'
import { useRouter } from 'next/navigation'
import { FC } from 'react'

const BillingPage: FC = () => {
  const { organization } = useAppContext()
  const { usage } = useUsage()
  const router = useRouter()
  if (!organization) return <></>

  if (isEnterprise(organization)) {
    // Enterprise users should not access this page
    router.push('/organization')
    return <></>
  }

  const billingCycle =
    usage?.current_period_start && usage?.current_period_end
      ? toDateCycle(usage.current_period_start, usage.current_period_end)
      : ''

  return (
    <PageLayout>
      <Head>
        <title>Billing - Dataherald AI API</title>
      </Head>
      <div className="flex flex-col gap-10 p-6">
        <div className="flex flex-col gap-5 max-w-2xl">
          <h1 className="text-xl font-bold">Pay as you go</h1>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <h2 className="pb-1 font-bold text-lg">Pending invoice</h2>
              <TooltipProvider>
                <Tooltip delayDuration={0}>
                  <TooltipTrigger asChild>
                    <Info size={14} />
                  </TooltipTrigger>
                  <TooltipContent className="bg-slate-600 border-slate-600">
                    <span className="text-white">
                      {`This is your current billing cycle's API usage, minus any
                      credits granted to you.`}
                    </span>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
            <div className="flex items-end gap-2">
              {usage ? (
                <>
                  <div className="text-3xl">
                    ${toDollars(usage?.amount_due)}
                  </div>
                  <span className="pb-1 text-xs text-slate-500">
                    {billingCycle}
                  </span>
                </>
              ) : (
                <Skeleton className="w-40 h-9" />
              )}
            </div>
            <span className="text-slate-700">
              {`You'll be billed at the end of your billing cycle for the usage during that cycle.`}
            </span>
          </div>
        </div>
        <ContentBox className="max-w-2xl">
          <PaymentMethodsList />
        </ContentBox>
      </div>
    </PageLayout>
  )
}

export default BillingPage
