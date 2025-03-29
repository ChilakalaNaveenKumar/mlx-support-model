
// React component for user profile
import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { fetchUserData, updateUserProfile } from '../api/userService';
import ProfileHeader from './ProfileHeader';
import ActivityFeed from './ActivityFeed';

const UserProfile = () => {
  const { userId } = useParams();
  const [userData, setUserData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  useEffect(() => {
    // Fetch user data when component mounts
    const loadUserData = async () => {
      try {
        setLoading(true);
        const data = await fetchUserData(userId);
        setUserData(data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load user data');
        setLoading(false);
      }
    };
    
    loadUserData();
  }, [userId]);
  
  const handleProfileUpdate = async (updatedData) => {
    try {
      const result = await updateUserProfile(userId, updatedData);
      if (result.success) {
        setUserData({...userData, ...updatedData});
        return true;
      }
      return false;
    } catch (err) {
      setError('Failed to update profile');
      return false;
    }
  };
  
  if (loading) {
    return <div>Loading profile...</div>;
  }
  
  if (error) {
    return <div className="error-message">{error}</div>;
  }
  
  return (
    <div className="user-profile-container">
      <ProfileHeader 
        username={userData.username}
        avatarUrl={userData.avatarUrl}
        joinDate={userData.joinDate}
      />
      
      <div className="profile-content">
        <div className="user-stats">
          <div className="stat-item">
            <span className="stat-value">{userData.postCount}</span>
            <span className="stat-label">Posts</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{userData.followerCount}</span>
            <span className="stat-label">Followers</span>
          </div>
        </div>
        
        <ActivityFeed userId={userId} />
      </div>
    </div>
  );
};

export default UserProfile;
