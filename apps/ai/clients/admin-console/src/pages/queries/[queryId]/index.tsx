import QueryLayout from '@/components/query/layout'
import LoadingQuery from '@/components/query/loading'
import QueryWorkstation from '@/components/query/workstation'
import { useQuery } from '@/hooks/api/useQuery'
import { Query } from '@/models/api'
import { useRouter } from 'next/router'
import { FC } from 'react'

const QueryPage: FC = () => {
  const router = useRouter()
  const { queryId } = router.query

  const { query, isLoading, error } = useQuery(Number(queryId))

  let pageContent: JSX.Element

  if (isLoading && !query) pageContent = <LoadingQuery />
  else if (error) pageContent = <div>Error loading the query</div>
  else pageContent = <QueryWorkstation query={query as Query} />

  return <QueryLayout>{pageContent}</QueryLayout>
}

export default QueryPage
