import { redirect } from 'next/navigation';

export default async function LocalRootPage({
  params
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  redirect(`/${locale}/upload`);
}
