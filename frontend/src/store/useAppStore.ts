import {create} from 'zustand'

interface AppState {
  // Upload flow state
  currentDatasetId: string | null
  uploadedFilePath: string | null
  suggestedTarget: string | null
  confirmedTarget: string | null
  profiles: any[]
  compositeTarget: any | null

  // Training state
  currentJobId: string | null
  trainingStatus: string | null

  // Actions
  setDatasetId: (id: string) => void
  setUploadedFilePath: (path: string) => void
  setSuggestedTarget: (target: string) => void
  setProfiles: (profiles: any[]) => void
  setCompositeTarget: (config: any) => void
  setJobId: (id: string) => void
  setTrainingStatus: (status: string) => void
  reset: () => void
}

const initialState = {
  currentDatasetId: null,
  uploadedFilePath: null,
  suggestedTarget: null,
  confirmedTarget: null,
  profiles: [],
  compositeTarget: null,
  currentJobId: null,
  trainingStatus: null
}

export const useAppStore = create<AppState>(
  (set) => ({
    ...initialState,
    setDatasetId: (id) =>
      set({currentDatasetId: id}),
    setUploadedFilePath: (path) =>
      set({uploadedFilePath: path}),
    setSuggestedTarget: (target) =>
      set({suggestedTarget: target}),
    setProfiles: (profiles) =>
      set({profiles}),
    setCompositeTarget: (config) =>
      set({compositeTarget: config}),
    setJobId: (id) =>
      set({currentJobId: id}),
    setTrainingStatus: (status) =>
      set({trainingStatus: status}),
    reset: () => set(initialState)
  })
)
