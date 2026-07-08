import axios from 'axios'

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000'

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api/v1`,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': process.env.NEXT_PUBLIC_API_KEY || 'dev-key'
  },
  timeout: 30000
})

// Response interceptor cho error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:',
      error.response?.data || error.message)
    return Promise.reject(error)
  }
)
